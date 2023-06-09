#!/usr/bin/env python3

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.objects.shape import Shape
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

SCENE_FILE = join(dirname(abspath(__file__)), 
                  '../../scenes/CSROS2_10_RL_2_cubes.ttt')

POS_MIN, POS_MAX = [-1.7250e+00, +7.0000e-01, +7.0200e-01], [-1.3250e+00, +7.0000e-01, +7.0200e-01]
EPISODES = 5
EPISODE_LENGTH = 200

# Workspace coordinates for cubes
[x_min, y_min, z_min] = [-1.7000e+00, +1.5000e-01, +7.2500e-01]
[x_max, y_max] = [-1.3250e+00, +6.0000e-01]

# Adding a replay buffer class to store experiences
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    

# Add missing functions
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def to_tensor(ndarray, requires_grad=False):
    return torch.tensor(ndarray, dtype=torch.float, device=device, requires_grad=requires_grad)



class PusherEnv(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.red_cube = Shape('Cuboid_red')
        self.blue_cube = Shape('Cuboid_blue')
        self.target_plane = Shape('Plane')  # assume the white plane is named 'Plane'
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    CUBE_POS_MIN, CUBE_POS_MAX = [x_min, y_min, z_min], [x_max, y_max, z_min]  # z_min and z_max are the same
    
    def _get_state(self):
        # Return state containing arm joint angles/velocities & position of both cubes and the target plane
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.red_cube.get_position(),
                               self.blue_cube.get_position(),
                               self.target_plane.get_position()])

    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target_plane.set_position(pos)

        # Set random positions for the cubes
        cube_pos_red = list(np.random.uniform(self.CUBE_POS_MIN, self.CUBE_POS_MAX))
        cube_pos_blue = list(np.random.uniform(self.CUBE_POS_MIN, self.CUBE_POS_MAX))
        self.red_cube.set_position(cube_pos_red)
        self.blue_cube.set_position(cube_pos_blue)

        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()
        
    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target_plane.get_position()
        # Reward is negative distance to target
        reward = -np.min([np.linalg.norm(np.array([ax, ay, az]) - np.array(self.red_cube.get_position())),
                          np.linalg.norm(np.array([ax, ay, az]) - np.array(self.blue_cube.get_position()))])
        return reward, self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Agent(object):

    def __init__(self, nb_states, nb_actions, args, replay_buffer):

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
        'hidden1':args.hidden1, 
        'hidden2':args.hidden2, 
        'init_w':args.init_w
        }  
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        self.replay_buffer = replay_buffer


        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

    
    def act(self, state):
        # Sample action from the policy
        action = self.actor(state)
        action = action.data
        self.a_t = action
        return action

    def observe(self, r_t, s_t1, done):
        # Store transition in the replay buffer.
        self.memory.append(self.s_t, self.a_t, r_t, s_t1, done)
        self.s_t = s_t1

    def random_action(self):
        # Returns a random action
        action = torch.Tensor([np.random.uniform(-1.,1.,self.nb_actions)]) # each action is a vector
        self.a_t = action
        return action

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
        to_tensor(next_state_batch, volatile=True),
        self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
        self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
        to_tensor(state_batch),
        self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def reset(self, obs):
        self.s_t = obs

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
        self.memory.cuda()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()


# Define the Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.relu(self.layer_1(torch.cat([x, u], 1)))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


replay_buffer = ReplayBuffer(10000)

env = PusherEnv()

class Config:
    def __init__(self):
        self.hidden1 = 400
        self.hidden2 = 300
        self.init_w = 0.003
        self.prate = 0.001
        self.rate = 0.001
        self.bsize = 64
        self.tau = 0.001
        self.discount = 0.99
        self.epsilon = 10000

config = Config()
agent = Agent(23, 7, config, replay_buffer)

for e in range(EPISODES):
    print('Starting episode %d' % e)
    state = env.reset()
    for i in range(EPISODE_LENGTH):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.observe(reward, next_state, done)
        if len(agent.memory) > agent.batch_size:
            agent.update_policy()
        if done:
            state = env.reset()
        else:
            state = next_state

print('Done!')
env.shutdown()
