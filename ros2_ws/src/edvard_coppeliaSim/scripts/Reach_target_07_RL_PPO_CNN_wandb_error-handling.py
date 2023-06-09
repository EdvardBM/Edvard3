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

#WandB
import wandb
wandb.login()

SCENE_FILE = join(dirname(abspath(__file__)), 
                  '../../scenes/CSROS2_1reach_target_RL.ttt')

# Waypoint pos min / max
POS_MIN, POS_MAX = [-1.7500e+00, +1.2500e-01, +7.0200e-01], [-1.2750e+00, +8.5000e-01, +7.0200e-01]

EPISODES = 5
EPISODE_LENGTH = 200

wandb.init(project="Reach_target_Run_01",
           config={
               "learning_rate": 1e-3,
               "episodes": EPISODES,
               "episode_length": EPISODE_LENGTH
           })
best_reward = -np.inf


joint_dim = 7 * 2  # joint positions and velocities
waypoint_dim = 3
state_dim = joint_dim + waypoint_dim  # Adjust as needed based on your specific robot and task
action_dim = 7
image_dim = (4, 128, 128)  # 3 channels for RGB, 1 channel for depth, 128x128 pixels

class PPO:
    def __init__(self, policy, optimizer, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def update(self, states, actions, images, log_probs_old, returns, advantages):
        for _ in range(self.ppo_epochs):
            for index in range(0, len(states), self.mini_batch_size):
                states_batch = states[index:index+self.mini_batch_size]
                images_batch = images[index:index+self.mini_batch_size]  # Add this line
                actions_batch = actions[index:index+self.mini_batch_size]
                log_probs_old_batch = log_probs_old[index:index+self.mini_batch_size]
                returns_batch = returns[index:index+self.mini_batch_size]
                advantages_batch = advantages[index:index+self.mini_batch_size]
                dist, value = self.policy(states_batch, images_batch)  # Pass images_batch to policy
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
    def __init__(self, state_dim, action_dim, image_dim):
        super(Policy, self).__init__()
        self.action_dim = action_dim

        # Define the CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(image_dim[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (image_dim[1]//4) * (image_dim[2]//4), 256), 
            nn.ReLU()
        )

        # Define the rest of the network for state processing
        self.actor = nn.Sequential(
            nn.Linear(state_dim + 256, 64),  # state_dim + 256 (output of the cnn)
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim + 256, 64),  # state_dim + 256 (output of the cnn)
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state, image):
        image = image.view(-1, *image_dim)  # reshape the image to the format [batch, channels, height, width]
        batch_size = image.shape[0]  # get the batch size from the image tensor

        image_processed = self.cnn(image)  # pass the image through the cnn
        # print('State: ' + str(state.size()))
        # print('Image Processed: ' + str(image_processed.size()))
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # adds an extra dimension, making state a 2D tensor with shape [1, 17]
            state = state.repeat(batch_size, 1)  # repeat the state tensor for batch_size times
        x = torch.cat((state, image_processed), dim=1)  # concatenate the processed image and state

        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean)
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        value = self.critic(x)
    
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
        image_depth = np.expand_dims(image_depth, -1)  # adds an extra dimension

        image = np.concatenate((image_rgb, image_depth), axis=-1)
        state = np.concatenate([self.agent.get_joint_positions(),
                            self.agent.get_joint_velocities(),
                            self.waypoint.get_position()])
        return state, image

    def reset(self):
        # Get a random position within a cuboid and set the waypoint position
        pos = list(np.random.uniform(self.WAYPOINT_POS_MIN, self.WAYPOINT_POS_MAX))
        self.waypoint.set_position(pos)

        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()
        
    def step(self, action):
        try:
            self.agent.set_joint_target_velocities(action)  # Execute action on arm
            self.pr.step()  # Step the physics simulation
            self.vision_sensor.handle_explicitly()
            ax, ay, az = self.agent_ee_tip.get_position()
            wx, wy, wz = self.waypoint.get_position()
            # Reward is negative distance to target
            reward = -np.linalg.norm(np.array([ax, ay, az]) - np.array([wx, wy, wz]))
            done = reward > -0.01  # if the arm's tip is close enough to the waypoint
        except Exception as e:
            print(f"An error occurred during simulation step: {e}")
            reward = -np.inf  # or some other default reward
            done = True  # end the episode if an error occurs
        finally:
            return reward, self._get_state(), done

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

class Agent(object):
    def __init__(self):
        self.policy = Policy(state_dim, action_dim, image_dim)
        lr = wandb.config.learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.ppo = PPO(self.policy, self.optimizer)


    def act(self, state, image):
        state = torch.tensor(state, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32)
        with torch.no_grad():
            dist, value = self.policy(state, image)
        action = dist.sample() 
        log_prob = dist.log_prob(action)
        return action.numpy().squeeze(), value.item(), log_prob.item()

    def learn(self, states, actions, log_probs_old, rewards, masks, values, images):  # Add images parameter
    
        states = torch.tensor(np.array(states), dtype=torch.float32)
        images = torch.tensor(np.array(images), dtype=torch.float32)
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
        self.ppo.update(states, actions, images, log_probs_old, returns, advantages)  # Add images parameter

env = ReachEnv()
agent = Agent()

for e in range(EPISODES):
    print('Starting episode %d' % e)
    state, image = env.reset()
    states = []
    images = []
    actions = []
    log_probs_old = []
    values = []
    rewards = []
    masks = []
    for i in range(EPISODE_LENGTH):
        action, value, log_prob = agent.act(state, image)
        states.append(state)
        images.append(image)  # Append image to images
        actions.append(action)
        log_probs_old.append(log_prob)
        values.append(value)
        reward, (next_state, next_image), done = env.step(action)
        rewards.append(reward)
        masks.append(not done)
        state = next_state
        image = next_image
    agent.learn(states, actions, log_probs_old, rewards, masks, values, images)
    episode_reward = sum(rewards)
    wandb.log({"episode_reward": episode_reward})

    # Save the model if it's the best so far
    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(agent.policy.state_dict(), 'best_model.pth')
        wandb.save('best_model.pth')

print('Done!')
wandb.finish()