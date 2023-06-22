#!/usr/bin/env python3

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
from pathlib import Path
import glob
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

import os
import shutil
from pathlib import Path

SCENE_FILE = join(dirname(abspath(__file__)), 
                  '../../../scenes/Kuka_Reach_target_constraints.ttt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 20000
EPISODE_LENGTH = 200
LAST_EPISODES_MEMORY = 3
SAVE_EVERY_X_EPISODE = 100
LR = 1e-4

USE_WANDB = True
HEADLESS = False 
BASE_DIR = 'End_game'
RUN_NAME = 'Run_GPU_10'
LOAD_PREVIOUS_RUN = True
EPISODE_NUM = 'Episode_2206'
PREVIOUS_RUN_PATH = f"/root/{BASE_DIR}/{RUN_NAME}/{EPISODE_NUM}/model.pth"

WANDB_CONTINUE = LOAD_PREVIOUS_RUN 
WANDB_PROJECT_NAME = BASE_DIR
run = None

joint_dim = 7 * 2  
waypoint_dim = 3
state_dim = joint_dim + waypoint_dim  
action_dim = 7
image_dim = (4, 128, 128) 

class PPO:
    def __init__(self, policy, optimizer, device, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64):
        self.policy = policy
        self.optimizer = optimizer
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def update(self, states, actions, images, log_probs_old, returns, advantages):
        # Move all your inputs to the GPU
        states = states.to(self.device)
        actions = actions.to(self.device)
        images = images.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

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

                # If you plan to use any of the outputs outside of the GPU, move them back to CPU
                if USE_WANDB:
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
        self.pr.launch(SCENE_FILE, headless=HEADLESS)
        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.waypoint = Shape('Waypoint')
        self.vision_sensor = VisionSensor('Vision_sensor')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.time_inside_target = 0.0
        self.time_outside_target = 0.0

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
        self.time_outside_target = 0.0
        pos = list(np.random.uniform(self.WAYPOINT_POS_MIN, self.WAYPOINT_POS_MAX))
        self.waypoint.set_position(pos)

        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()
        
    JOINT_LIMITS = {
        'joint1': tuple(np.radians([-60.0, 60.0])),
        'joint2': tuple(np.radians([-1.0e-03, 130.0])),
        'joint3': tuple(np.radians([-85.0, 85.0])),
        'joint4': tuple(np.radians([-120.0, 120.0])),
        'joint5': tuple(np.radians([-85.0, 85.0])),
        'joint6': tuple(np.radians([-120.0, 120.0])),
        'joint7': tuple(np.radians([-175.0, 175.0])),
    }
    
    TARGET_THRESHOLD = 0.01  

    USE_TIME_PENALTY = True 
    TIME_PENALTY = -1.0 

    USE_LIMIT_PENALTY = True
    LIMIT_PENALTY = -10.0
    
    USE_INSIDE_TARGET_REWARD = False
    INSIDE_TARGET_REWARD = 1.0
    TIME_INSIDE_TARGET_REWARD_INCREMENT = 0.2

    USE_DISTANCE_PENALTY = True   
    DISTANCE_PENALTY = -1.0      

    DONE_ON_REWARD_THRESHOLD = False
    REWARD_THRESHOLD = -0.001

    def step(self, action, e, best_episode):
        try:
            self.agent.set_joint_target_velocities(action)  
            self.pr.step() 
            self.vision_sensor.handle_explicitly()
            ax, ay, az = self.agent_ee_tip.get_position()
            wx, wy, wz = self.waypoint.get_position()
            distance_to_target = np.linalg.norm(self.agent_ee_tip.get_position() - self.waypoint.get_position())

            reward = 0

            if self.USE_TIME_PENALTY and distance_to_target > self.TARGET_THRESHOLD:
                self.time_outside_target += 1
                reward += self.TIME_PENALTY

            if self.USE_DISTANCE_PENALTY:
                reward += self.DISTANCE_PENALTY * np.linalg.norm(np.array([ax, ay, az]) - np.array([wx, wy, wz]))

            if self.USE_INSIDE_TARGET_REWARD:
                if distance_to_target < self.TARGET_THRESHOLD:
                    self.time_inside_target += 1  
                    reward += self.INSIDE_TARGET_REWARD + self.TIME_INSIDE_TARGET_REWARD_INCREMENT * self.time_inside_target
                else:
                    self.time_inside_target = 0.0 

            if self.USE_LIMIT_PENALTY:
                joint_positions = self.agent.get_joint_positions()
                for idx, joint_position in enumerate(joint_positions):
                    joint_limit = self.JOINT_LIMITS[f'joint{idx + 1}']
                    if not (joint_limit[0] <= joint_position <= joint_limit[1]):
                        reward += self.LIMIT_PENALTY
                        break
                    
            done = self.DONE_ON_REWARD_THRESHOLD and reward > self.REWARD_THRESHOLD
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
        self.policy = self.policy.to(device)
        lr = LR
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.ppo = PPO(self.policy, self.optimizer, device) 

    def act(self, state, image):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        image = torch.tensor(image, dtype=torch.float32).to(device)
        with torch.no_grad():
            dist, value = self.policy(state, image)
        action = dist.sample() 
        log_prob = dist.log_prob(action)
        return action.cpu().numpy().squeeze(), value.cpu().item(), log_prob.cpu().item()

    def learn(self, states, actions, log_probs_old, rewards, masks, values, images):  
    
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        images = torch.tensor(np.array(images), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        masks = torch.tensor(np.array(masks), dtype=torch.float32).to(device)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(device)

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + 0.99 * R * masks[t]
            returns[t] = R
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - values
        self.ppo.update(states, actions, images, log_probs_old, returns, advantages) 

class RunManager:
    def __init__(self, base_dir, run_name):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.run_dir = self.base_dir / run_name
        self.best_reward = -np.inf
        self.best_episode_dir = None
        self.best_episodes = []

    def create_directory(self, directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"Unable to create directory {directory}. {e}")

    def create_directories(self, load_previous_run):
        self.create_directory(self.base_dir)
        if not load_previous_run and self.run_dir.exists():
            self.run_name = self.create_unique_run_name()
            self.run_dir = self.base_dir / self.run_name
        self.create_directory(self.run_dir)
        self.script_info_path = self.run_dir / 'script_info.txt'

    def create_unique_run_name(self):
        i = 1
        while True:
            new_run_name = f"{self.run_name}_{i}"
            if not (self.base_dir / new_run_name).exists():
                return new_run_name
            i += 1

    def update_best_episode(self, episode_reward, e, episode_dir, agent):
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_episode = e
            print(f"New best episode found! Episode: {self.best_episode}, Reward: {self.best_reward}")
            if self.best_episode_dir is not None:
                shutil.rmtree(self.best_episode_dir, ignore_errors=True)
            self.best_episode_dir = os.path.join(self.run_dir, 'best_episode')
            if os.path.exists(self.best_episode_dir):
                shutil.rmtree(self.best_episode_dir)
            shutil.copytree(episode_dir, self.best_episode_dir)
            self.best_episodes.append(e)
        return self.best_episode_dir

    def maintain_last_episodes_memory(self, episode):
            episode_dirs = sorted(glob.glob(str(self.run_dir / 'Episode_*')), key=lambda x: int(x.split('_')[-1]))
            while len(episode_dirs) > LAST_EPISODES_MEMORY:
                if int(episode_dirs[0].split('_')[-1]) % SAVE_EVERY_X_EPISODE != 0:  # Skip the episodes that should be saved
                    print(f"Deleting directory: {episode_dirs[0]}") 
                    shutil.rmtree(episode_dirs[0], ignore_errors=True)
                episode_dirs.pop(0)

    def load_wandb_id(self, previous_run_path=None):
        if previous_run_path:
            # If previous_run_path is provided, construct the path to script_info.txt
            script_info_path = os.path.join(os.path.dirname(previous_run_path), 'script_info.txt')
        else:
            # If previous_run_path is not provided, use the default script_info_path
            script_info_path = self.script_info_path

        if os.path.exists(script_info_path):
            with open(script_info_path, 'r') as f:
                for line in f:
                    if 'Wandb ID' in line:
                        return line.split(': ')[-1].strip()
        return None

env = ReachEnv()
agent = Agent()
run_manager = RunManager(base_dir=BASE_DIR, run_name=RUN_NAME)
run_manager.create_directories(load_previous_run=LOAD_PREVIOUS_RUN) 
states_history = [] 
best_reward = -np.inf
best_episode_dir = None
best_episodes = []
best_episode = -1

if LOAD_PREVIOUS_RUN:
    save_info = torch.load(PREVIOUS_RUN_PATH)
    agent.policy.load_state_dict(save_info['model_state_dict'])  # Load the model state
    agent.optimizer.load_state_dict(save_info['optimizer_state_dict'])  # Load the optimizer state
    run_manager.best_reward = save_info['best_reward']
    start_episode = save_info['episode'] + 1
    best_episode_dir = os.path.dirname(PREVIOUS_RUN_PATH)

    # Load other data as necessary
    states = save_info['other_data']['states']
    actions = save_info['other_data']['actions']
    log_probs_old = save_info['other_data']['log_probs_old']
    values = save_info['other_data']['values']
    masks = save_info['other_data']['masks']
    images = save_info['other_data']['images']
    depth_images = save_info['other_data']['depth_images']
else:
    start_episode = 0
    best_episode_dir = run_manager.run_dir / 'best_episode'
    best_episode_dir.mkdir(parents=True, exist_ok=True) 

script_info = f'Script: {__file__}\n'
script_info += f'Script Path: {os.path.abspath(__file__)}\n'
script_info_path = run_manager.run_dir / 'script_info.txt'
with open(script_info_path, 'w') as f:
    f.write(script_info)

if USE_WANDB:
    import wandb
    wandb.login()

    if WANDB_CONTINUE:
        wandb_id = run_manager.load_wandb_id(PREVIOUS_RUN_PATH)

    if WANDB_CONTINUE and wandb_id: 
        run = wandb.init(id=wandb_id,
                         resume="Force",
                         project=WANDB_PROJECT_NAME,
                         save_code=True)
    else:
        wandb_id = wandb.util.generate_id()
        run = wandb.init(project=WANDB_PROJECT_NAME,
                         id=wandb_id,
                         save_code=True,
                         config={
                             "learning_rate": LR,
                             "episodes": EPISODES,
                             "episode_length": EPISODE_LENGTH
                         })
    if run:
        wandb_id = run.id
        script_info += f'Wandb ID: {wandb_id}\n' 
        with open(run_manager.script_info_path, 'w') as f:  
            f.write(script_info)

            
for e in range(start_episode, EPISODES):
    print('Starting episode %d' % e)
    state, image = env.reset()
    states = []
    images = []
    depth_images = []
    actions = []
    log_probs_old = []
    values = []
    rewards = []
    masks = []
    actions_history = [] 
    action_log_probs = []
    total_distance = 0
    last_distance = 0
    states_history.append(state) 
    episode_dir = os.path.join(run_manager.run_dir, f'Episode_{e}')
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
        depth_images.append(image[:, :, 3])
        actions_history.append(action)
        distance_to_target = np.linalg.norm(env.agent_ee_tip.get_position() - env.waypoint.get_position())
        total_distance += distance_to_target
        last_distance = distance_to_target
    action_log_probs = np.array(action_log_probs)
    agent.learn(states, actions, log_probs_old, rewards, masks, values, images)
    episode_reward = sum(rewards)
    average_reward = episode_reward / len(rewards)
    average_distance = total_distance / EPISODE_LENGTH
    if USE_WANDB:
        wandb.log({
            "Episode Reward": episode_reward, 
            "Average Reward": average_reward,
            "Average Distance": average_distance,
            "Last Distance": last_distance
        })
    save_info = {
        'episode': e,
        'model_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'best_reward': run_manager.best_reward,
        'other_data': {
            'states': states,
            'actions': actions,
            'log_probs_old': log_probs_old,
            'values': values,
            'masks': masks,
            'images': images,
            'depth_images': depth_images
        }
    }
    torch.save(save_info, os.path.join(episode_dir, 'model.pth'))
    np.save(os.path.join(episode_dir, 'states.npy'), states)
    np.save(os.path.join(episode_dir, 'actions.npy'), actions)
    np.save(os.path.join(episode_dir, 'log_probs_old.npy'), log_probs_old)
    np.save(os.path.join(episode_dir, 'values.npy'), values)
    np.save(os.path.join(episode_dir, 'masks.npy'), masks)
    np.save(os.path.join(episode_dir, 'images.npy'), images)
    np.save(os.path.join(episode_dir, 'depth_images.npy'), depth_images)
    shutil.copy(os.path.join(run_manager.run_dir, 'script_info.txt'), episode_dir)
    best_episode_dir = run_manager.update_best_episode(episode_reward, e, episode_dir, agent)
    run_manager.maintain_last_episodes_memory(e)

BASE_DIR = Path(BASE_DIR)
RUN_NAME = Path(RUN_NAME)
BEST_EPISODE_DIR = Path(best_episode_dir)
'''
make_video_command = join(dirname(abspath(__file__)), 'commands', 'make_video.py')
subprocess.run([make_video_command, f'{BEST_EPISODE_DIR}/images.npy'])
subprocess.run([make_video_command, f'{BEST_EPISODE_DIR}/depth_images.npy'])
'''
print('Done!')
env.shutdown()

if USE_WANDB:
    wandb.finish()