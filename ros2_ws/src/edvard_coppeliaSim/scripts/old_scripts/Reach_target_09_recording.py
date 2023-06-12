#!/usr/bin/env python3

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

import wandb
wandb.login()

import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


SCENE_FILE = join(dirname(abspath(__file__)), 
                  '../../scenes/CSROS2_1reach_target_RL.ttt')

EPISODES = 5
EPISODE_LENGTH = 200
BEST_EPISODES_MEMORY = 2

wandb.init(project="Reach_target_Run_01",
           config={
               "learning_rate": 1e-3,
               "episodes": EPISODES,
               "episode_length": EPISODE_LENGTH
           })

joint_dim = 7 * 2  
waypoint_dim = 3
state_dim = joint_dim + waypoint_dim  
action_dim = 7
image_dim = (4, 128, 128) 

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
                images_batch = images[index:index+self.mini_batch_size]  
                actions_batch = actions[index:index+self.mini_batch_size]
                log_probs_old_batch = log_probs_old[index:index+self.mini_batch_size]
                returns_batch = returns[index:index+self.mini_batch_size]
                advantages_batch = advantages[index:index+self.mini_batch_size]
                dist, value = self.policy(states_batch, images_batch)  
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
                wandb.log({"Actor Loss": actor_loss.item(), "Critic Loss": critic_loss.item(), "Total Loss": loss.item()})

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, image_dim):
        super(Policy, self).__init__()
        self.action_dim = action_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(image_dim[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (image_dim[1]//4) * (image_dim[2]//4), 256), 
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(state_dim + 256, 64),  
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim + 256, 64),  
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state, image):
        image = image.view(-1, *image_dim) 
        batch_size = image.shape[0]  

        image_processed = self.cnn(image) 
        if len(state.shape) == 1:
            state = state.unsqueeze(0) 
            state = state.repeat(batch_size, 1) 
        x = torch.cat((state, image_processed), dim=1) 

        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean)
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        value = self.critic(x)
    
        return dist, value

class ReachEnv(object):
    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=True)
        self.pr.start()
        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.waypoint = Shape('Waypoint')
        self.vision_sensor = VisionSensor('Vision_sensor')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    WAYPOINT_POS_MIN, WAYPOINT_POS_MAX = [-1.7500e+00, +1.2500e-01, +7.0200e-01], [-1.2750e+00, +8.5000e-01, +7.0200e-01]
    
    def _get_state(self):
        image_rgb = self.vision_sensor.capture_rgb()
        image_depth = self.vision_sensor.capture_depth()
        image_depth = np.expand_dims(image_depth, -1) 

        image = np.concatenate((image_rgb, image_depth), axis=-1)
        state = np.concatenate([self.agent.get_joint_positions(),
                            self.agent.get_joint_velocities(),
                            self.waypoint.get_position()])
        return state, image
    
    def reset(self):
        self.pr.stop()
        self.pr.start()

        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.waypoint = Shape('Waypoint')
        self.vision_sensor = VisionSensor('Vision_sensor')
        self.agent_ee_tip = self.agent.get_tip()

        pos = list(np.random.uniform(self.WAYPOINT_POS_MIN, self.WAYPOINT_POS_MAX))
        self.waypoint.set_position(pos)

        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()
        
    def step(self, action, e, best_episode):
        try:
            self.agent.set_joint_target_velocities(action)  
            self.pr.step() 
            self.vision_sensor.handle_explicitly()
            ax, ay, az = self.agent_ee_tip.get_position()
            wx, wy, wz = self.waypoint.get_position()
            reward = -np.linalg.norm(np.array([ax, ay, az]) - np.array([wx, wy, wz]))
            done = reward > -0.001  
        except Exception as e:
            print(f"An error occurred during simulation step: {e}")
            reward = -np.inf
            done = True 
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

class RunManager:
    def __init__(self, base_dir, run_name):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.run_dir = self.base_dir / run_name

    def prompt_for_overwrite(self, directory):
        while True:
            response = input(f'Directory {directory} already exists. Overwrite? (y/n): ')
            if response.lower() in ['y', 'n']:
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        if response.lower() == 'y':
            try:
                shutil.rmtree(directory)
                print(f"Directory {directory} removed.")
                self.create_directories()  # remove argument here
            except OSError as e:
                print(f"Unable to remove directory {directory}. {e}")
        else:
            print(f"Directory {directory} not removed.")

    def move_best_model(self):
        source_path = self.run_dir / 'best_episode' / 'best_model.pth'
        target_path = self.run_dir / 'best_parameters' / 'best_model.pth'
        shutil.move(source_path, target_path)

    def create_directory(self, directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"Unable to create directory {directory}. {e}")

    def create_directories(self):
        self.create_directory(self.base_dir)
        if self.run_dir.exists():
            self.prompt_for_overwrite(self.run_dir)
        else:
            self.create_directory(self.run_dir)

        subdirs = ['Video', 'data', 'best_parameters']
        for subdir in subdirs:
            subdir_path = self.run_dir / subdir
            self.create_directory(subdir_path)
 
    def make_video(self):
        import glob
        from moviepy.editor import ImageSequenceClip

        # Load images
        images = np.load(f'{self.run_dir}/best_episode/images.npy')

       # Save each image as a separate PNG file
        for idx, img in enumerate(images):
            plt.imsave(f'{self.run_dir}/best_episode/{idx:05d}.png', img)

        # List image files in order
        image_files = sorted(glob.glob(f'{self.run_dir}/best_episode/*.png'), key=lambda x: int(os.path.basename(x).split('.')[0]))

        # Create a video from the image sequence
        clip = ImageSequenceClip(image_files, fps=24)
    
    # Make sure the output directory exists
        output_dir = os.path.join(self.run_dir, 'Video')
        os.makedirs(output_dir, exist_ok=True)

    # Save video
        clip.write_videofile(f'{output_dir}/best_run.mp4')

    # Delete the image files
        for file in image_files:
            os.remove(file)


BASE_DIR = 'Reach_target_Recordings'
RUN_NAME = 'Run_01'
run_manager = RunManager(base_dir=BASE_DIR, run_name=RUN_NAME)
run_manager.create_directories() 
states_history = [] 

env = ReachEnv()
agent = Agent()
best_reward = -np.inf
best_episode_dir = None
best_episodes = []
best_episode = -1

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
    actions_history = [] 
    action_log_probs = []
    states_history.append(state) 
    episode_dir = os.path.join(run_manager.run_dir, f'Episode_{e}')
    os.makedirs(episode_dir, exist_ok=True)
    np.save(os.path.join(episode_dir, 'states.npy'), states_history)
    np.save(os.path.join(episode_dir, 'actions.npy'), actions_history)
    np.save(os.path.join(episode_dir, 'images.npy'), images)
    np.save(os.path.join(episode_dir, 'rewards.npy'), rewards)
    os.makedirs(episode_dir, exist_ok=True)
    for i in range(EPISODE_LENGTH):
        action, value, log_prob = agent.act(state, image)
        states.append(state)
        images.append(image)  
        actions.append(action)
        log_probs_old.append(log_prob)
        values.append(value)
        reward, (next_state, next_image), done = env.step(action, e, best_episode if best_episode >= 0 else e)        
        rewards.append(reward)
        masks.append(not done)
        state = next_state
        image = next_image
        actions_history.append(action) 

    action_log_probs = np.array(action_log_probs)
    fig, ax = plt.subplots()
    ax.hist(action_log_probs, bins='auto')  
    ax.set_title("Action distribution")
    wandb.log({"Action distribution": wandb.Image(fig)})
    plt.close(fig)  

    agent.learn(states, actions, log_probs_old, rewards, masks, values, images)
    episode_reward = sum(rewards)
    average_reward = episode_reward / len(rewards)
    wandb.log({"Episode Reward": episode_reward, "Average Reward": average_reward})
    np.save(os.path.join(episode_dir, 'states.npy'), states_history)
    np.save(os.path.join(episode_dir, 'actions.npy'), actions_history)
    np.save(os.path.join(episode_dir, 'images.npy'), images)
    np.save(os.path.join(episode_dir, 'rewards.npy'), rewards)


    if episode_reward > best_reward:
        best_reward = episode_reward
        best_episode = e
        if best_episode_dir is not None:
            shutil.rmtree(best_episode_dir, ignore_errors=True)
        best_episode_dir = os.path.join(run_manager.run_dir, 'best_episode')
        if os.path.exists(best_episode_dir):
            shutil.rmtree(best_episode_dir)
        shutil.copytree(episode_dir, best_episode_dir)

        torch.save(agent.policy.state_dict(), os.path.join(best_episode_dir, 'best_model.pth'))

        best_episodes.append(e)

    while len(best_episodes) > BEST_EPISODES_MEMORY:
        worst_episode = min(best_episodes, key=lambda episode: np.load(os.path.join(run_manager.run_dir, f'Episode_{episode}', 'rewards.npy')).sum())
        worst_episode_dir = os.path.join(run_manager.run_dir, f'Episode_{worst_episode}')
        shutil.rmtree(worst_episode_dir, ignore_errors=True)
        best_episodes.remove(worst_episode)

run_manager.move_best_model()
run_manager.make_video()
print('Done!')
wandb.finish()
env.shutdown()