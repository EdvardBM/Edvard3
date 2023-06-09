#!/usr/bin/env python3

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np

#PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

SCENE_FILE = join(dirname(abspath(__file__)), 
                  '../../scenes/CSROS2_1reach_target_RL.ttt')

# Waypoint pos min / max
POS_MIN, POS_MAX = [-1.7500e+00, +1.2500e-01, +7.0200e-01], [-1.2750e+00, +8.5000e-01, +7.0200e-01]
EPISODES = 5
EPISODE_LENGTH = 200

state_dim = 65553
action_dim = 7

class PPO:
    def __init__(self, policy, optimizer, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.ppo_epochs):
            for index in range(0, len(states), self.mini_batch_size):
                states_batch = states[index:index+self.mini_batch_size]
                actions_batch = actions[index:index+self.mini_batch_size]
                log_probs_old_batch = log_probs_old[index:index+self.mini_batch_size]
                returns_batch = returns[index:index+self.mini_batch_size]
                advantages_batch = advantages[index:index+self.mini_batch_size]
                dist, value = self.policy(states_batch)
                entropy = dist.entropy().mean()
                log_probs = dist.log_prob(actions_batch)
                ratio = (log_probs - log_probs_old_batch).exp()
                surrogate1 = ratio * advantages_batch
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_batch
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = 0.5 * (returns_batch - value).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        mean = self.actor(state)
        mean = self.actor(state).unsqueeze(0)
        std = self.log_std.exp().expand_as(mean)
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        value = self.critic(state)
        return dist, value

class ReachEnv(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.waypoint = Shape('Waypoint')
        self.vision_sensor = VisionSensor('Vision_sensor')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    # Define the minimum and maximum positions for the waypoint
    WAYPOINT_POS_MIN, WAYPOINT_POS_MAX = [-1.7250e+00, +7.0000e-01, +7.2500e-01], [-1.3250e+00, +7.0000e-01, +7.2500e-01]
    
    def _get_state(self):
        # Return state containing arm joint angles/velocities, position of the waypoint
        # and the vision sensor's captured image
        image_rgb = self.vision_sensor.capture_rgb()
        image_depth = self.vision_sensor.capture_depth()
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.waypoint.get_position(),
                               image_rgb.flatten(),
                               image_depth.flatten()])

    def reset(self):
        # Get a random position within a cuboid and set the waypoint position
        pos = list(np.random.uniform(self.WAYPOINT_POS_MIN, self.WAYPOINT_POS_MAX))
        self.waypoint.set_position(pos)

        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()
        
    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation

        self.vision_sensor.handle_explicitly()

        ax, ay, az = self.agent_ee_tip.get_position()
        wx, wy, wz = self.waypoint.get_position()
        # Reward is negative distance to target
        reward = -np.linalg.norm(np.array([ax, ay, az]) - np.array([wx, wy, wz]))
        done = reward > -0.01  # if the arm's tip is close enough to the waypoint
        return reward, self._get_state(), done

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

class Agent(object):
    def __init__(self):
        self.policy = Policy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.ppo = PPO(self.policy, self.optimizer)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            dist, value = self.policy(state)
        action = dist.sample() 
        log_prob = dist.log_prob(action)
        return action.numpy().squeeze(), value.item(), log_prob.item()

    def learn(self, states, actions, log_probs_old, rewards, masks, values):
        
        #Converts list appended in epsiode to tensor
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.float32)
        values = torch.tensor(np.array(values), dtype=torch.float32)


        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + 0.99 * R * masks[t]
            returns[t] = R
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - values
        self.ppo.update(states, actions, log_probs_old, returns, advantages)

env = ReachEnv()
agent = Agent()

for e in range(EPISODES):
    print('Starting episode %d' % e)
    state = env.reset()
    states = []
    actions = []
    log_probs_old = []
    values = []
    rewards = []
    masks = []
    for i in range(EPISODE_LENGTH):
        action, value, log_prob = agent.act(state)
        states.append(state)
        actions.append(action)
        log_probs_old.append(log_prob)
        values.append(value)
        reward, next_state, done = env.step(action)
        rewards.append(reward)
        masks.append(not done)
        state = next_state
    agent.learn(states, actions, log_probs_old, rewards, masks, values)
print('Done!')
env.shutdown()