#!/usr/bin/env python3

# Importing necessary libraries
from os.path import dirname, join, abspath  # Used for directory manipulations
from pyrep import PyRep  # PyRep is the Python interface for CoppeliaSim
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820  # Importing the Kuka 
from pyrep.objects.shape import Shape  # Used for the target, which is a shape in CoppeliaSim
from pyrep.objects.vision_sensor import VisionSensor  # The camera in CoppeliaSim
import numpy as np  # Used for array manipulations
from pathlib import Path  # Path manipulation library
import glob  # Used for file manipulations
import subprocess  # Used for running subprocesses, i.e. making video

# PyTorch library for neural networks, gradient descent, optimizer.
import torch
import torch.nn as nn  
import torch.optim as optim  
from torch.distributions import MultivariateNormal 

import os  # Used for interacting with the os
import shutil  # Used for editing files
from pathlib import Path  # Manipulate paths independent of the os

# Location of the CoppeliaSim scene file
SCENE_FILE = join(dirname(abspath(__file__)), '../../../scenes/Kuka_Reach_target_constraints.ttt')

# Setting device to CUDA if available on Idun, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 20000 # Number of episodes for the RL training
EPISODE_LENGTH = 200 # Length of each episode
LAST_EPISODES_MEMORY = 3 # How many of the last episodes are saved locally
SAVE_EVERY_X_EPISODE = 100 # The interval of locally saved episodes
LR = 1e-3 #Learning Rrate

USE_WANDB = True
HEADLESS = False # Flag for not GUI
BASE_DIR = 'End_game' # The directory in which the runs are saved
RUN_NAME = 'Run_GPU_14'
LOAD_PREVIOUS_RUN = True # Wether or not to load a previous run
EPISODE_NUM = 'Episode_1162' # Which episode to load
PREVIOUS_RUN_PATH = f"/root/{BASE_DIR}/{RUN_NAME}/{EPISODE_NUM}/model.pth"

WANDB_CONTINUE = LOAD_PREVIOUS_RUN 
WANDB_PROJECT_NAME = BASE_DIR
run = None #initialization of wandb run object

# Dimensions for state and action spaces
joint_dim = 7 * 2  # 7 joints in the Kuka arm, each with a position and velocity state
waypoint_dim = 3 # 3D target position
state_dim = joint_dim + waypoint_dim  
action_dim = 7 # The 7 joints to conntrol
image_dim = (4, 128, 128) #image dim, RGB-D (4) on 128x128 resolution

class PPO:
    def __init__(self, policy, optimizer, device, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64):
        self.policy = policy # Policy determines the actions taken by the agent
        self.optimizer = optimizer # Upddating the weights 
        self.device = device # CPU or GPU
        self.clip_epsilon = clip_epsilon # Parameter in for clipping used in the objective functiion 
        self.ppo_epochs = ppo_epochs # Number of iterations to update the policy network 
        self.mini_batch_size = mini_batch_size #Batch size for updating the policy network

    # Performs an update step using PPO onthe policy network
    def update(self, states, actions, images, log_probs_old, returns, advantages):
        # Moves the input data to GPU if it available
        states = states.to(self.device)
        actions = actions.to(self.device)
        images = images.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        for _ in range(self.ppo_epochs):
            #Mini-batch update starts here
            for index in range(0, len(states), self.mini_batch_size):
                 # Select a mini-batch of data
                states_batch = states[index:index+self.mini_batch_size]
                images_batch = images[index:index+self.mini_batch_size]
                actions_batch = actions[index:index+self.mini_batch_size]
                log_probs_old_batch = log_probs_old[index:index+self.mini_batch_size]
                returns_batch = returns[index:index+self.mini_batch_size]
                advantages_batch = advantages[index:index+self.mini_batch_size]
                # Get the distribution and value prediction from the policy network                
                dist, value = self.policy(states_batch, images_batch)  
                # Calculate the entropy of the distribution for exploration
                entropy = dist.entropy().mean()
                # Calculate the new log probabilities
                log_probs = dist.log_prob(actions_batch)
                # Calculate the ratio between new and old log probabilities
                ratio = (log_probs - log_probs_old_batch).exp()
                # Surrogate (objective) functions for the PPO ibjective
                surrogate1 = ratio * advantages_batch
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_batch
                # Actor loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                # Critic Loss
                critic_loss = 0.5 * (returns_batch - value).pow(2).mean()
                # Total loss with entropy regularization (encourage  exploring)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                # Perform a gradient update step
                self.optimizer.zero_grad() # Clear the gradients
                loss.backward()  # clac the gradients
                self.optimizer.step() # Update the weights

                if USE_WANDB:
                    wandb.log({"Actor Loss": actor_loss.item(), "Critic Loss": critic_loss.item(), "Total Loss": loss.item()})

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, image_dim):
        super(Policy, self).__init__()
        self.action_dim = action_dim

        # Define a CNN to process the input images.
        # It starts with two Convolutional layers with ReLU activations, followed by flattening the output,
        # and then a fully connected layer (Linear) also with ReLU activation.
        self.cnn = nn.Sequential(
            nn.Conv2d(image_dim[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (image_dim[1]//4) * (image_dim[2]//4), 256), 
            nn.ReLU()
        )

        # Define the actor network that determines the action to be taken. 
        # It is a Feed-Forward Neural Network with two hiddden layers and tanh activations.
        # The input is the combination of state information and the processed image information.
        self.actor = nn.Sequential(
            nn.Linear(state_dim + 256, 64),  
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )

        # Define the critic network that evaluates the value of a state-action pair.
        # The architectuere is similar to the actor network, but the output layer only has one "neuron"
        # because it is estimating a single value (state-value function).
        self.critic = nn.Sequential(
            nn.Linear(state_dim + 256, 64),  
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # The log standard deviation parameter for the action distribution. 
        # This will be learned during training.
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state, image):
         # Modifies the image to the correct dimensions for the CNN
        image = image.view(-1, *image_dim) 
        batch_size = image.shape[0]  
        # Pass the image through the CNN
        image_processed = self.cnn(image) 
        # If 1D tensor, make 2D tensor
        if len(state.shape) == 1:
            state = state.unsqueeze(0) 
            state = state.repeat(batch_size, 1) 
        # Concatenate the state and image_processed tensors along dimension 1
        x = torch.cat((state, image_processed), dim=1) 
        # The actor  calcs the mean of the action distribution
        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean) # Calcs standard deviation
        dist = MultivariateNormal(mean, torch.diag_embed(std)) # Multivariate Normal distribution
        # The critic calcs the state value     
        value = self.critic(x)
        # Return the action distribution and value
        return dist, value

class ReachEnv(object):
    def __init__(self):
        # Initialization
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=HEADLESS) #Launches CoppeliaSim scene
        # robot arm
        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False) # Disable the built-in control loop
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.waypoint = Shape('Waypoint') # Target
        self.vision_sensor = VisionSensor('Vision_sensor') # RGB-D camera
        self.agent_ee_tip = self.agent.get_tip() # dummy placed on tip of robot arm
        # Store the initial joint positions of the robot arm for resets
        self.initial_joint_positions = self.agent.get_joint_positions() #
        # time counters for rewards
        self.time_inside_target = 0.0
        self.time_outside_target = 0.0

    # Define boundaries for waypoint position, basically the table top
    WAYPOINT_POS_MIN, WAYPOINT_POS_MAX = [-1.7500e+00, +1.2500e-01, +7.0200e-01], [-1.2750e+00, +8.5000e-01, +7.0200e-01]
    
    # Get the current state of the environment.
    def _get_state(self):
        image_rgb = self.vision_sensor.capture_rgb()
        image_depth = self.vision_sensor.capture_depth()
        image_depth = np.expand_dims(image_depth, -1) # Expand dimensions of the depth image
        image = np.concatenate((image_rgb, image_depth), axis=-1) # Combine the RGB and depth images
        # Concatenate the joint positions, velocities, and waypoint position
        state = np.concatenate([self.agent.get_joint_positions(),
                            self.agent.get_joint_velocities(),
                            self.waypoint.get_position()])
        return state, image
    
    # Reset the environment to its initial state
    def reset(self):
        self.pr.stop()
        self.pr.start()

        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.waypoint = Shape('Waypoint')
        self.vision_sensor = VisionSensor('Vision_sensor')
        self.agent_ee_tip = self.agent.get_tip()
        self.time_outside_target = 0.0  # Reset the time outside target,
        # Randomly position the target within given boundaries
        pos = list(np.random.uniform(self.WAYPOINT_POS_MIN, self.WAYPOINT_POS_MAX))
        self.waypoint.set_position(pos)
        #reset agent
        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()
    
    # Reward control starting here
    # Define joint limits for the robot arm
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

    USE_TIME_PENALTY = False 
    TIME_PENALTY = -1.0 

    USE_LIMIT_PENALTY = True
    LIMIT_PENALTY = -10.0
    
    USE_INSIDE_TARGET_REWARD = True
    INSIDE_TARGET_REWARD = 5.0
    TIME_INSIDE_TARGET_REWARD_INCREMENT = 1

    USE_DISTANCE_PENALTY = True   
    DISTANCE_PENALTY = -1.0      

    DONE_ON_REWARD_THRESHOLD = False
    REWARD_THRESHOLD = -0.001
    # Reward control stopping here
    # Step in the env; apply action, return new state, reeward, and done flag
    def step(self, action, e, best_episode):
        try:
            # Set the wanted velocities based on the action
            self.agent.set_joint_target_velocities(action)  
            self.pr.step() # Take step in sim
            self.vision_sensor.handle_explicitly() # update vision sensor
            # calc distance to trrget 
            ax, ay, az = self.agent_ee_tip.get_position()
            wx, wy, wz = self.waypoint.get_position()
            distance_to_target = np.linalg.norm(self.agent_ee_tip.get_position() - self.waypoint.get_position())

            # Rewards starts here
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
            # Check if the episode is done based on the reward threshold     
            done = self.DONE_ON_REWARD_THRESHOLD and reward > self.REWARD_THRESHOLD
        except Exception as e:
            print(f"An error occurred during simulation step: {e}")
            reward = -np.inf # large negative reward for any error
            done = True 
        finally:
            return reward, self._get_state(), done # return NEW rward, state, and done flag
    # Close the sim
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

class Agent(object):
    def __init__(self):
        self.policy = Policy(state_dim, action_dim, image_dim) #Policy used to decide action
        self.policy = self.policy.to(device) # Move policy to GPU if in use
        lr = LR
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr) #Optimizer used to update the policy parameters
        self.ppo = PPO(self.policy, self.optimizer, device) #PPO use to train policy

    #Take action given the state and image
    def act(self, state, image):
        #Converts so it is possible to move to GPU
        state = torch.tensor(state, dtype=torch.float32).to(device)
        image = torch.tensor(image, dtype=torch.float32).to(device)
        # Policy give action dist and state value
        with torch.no_grad():
            dist, value = self.policy(state, image)
        action = dist.sample() # Sample an action fromdist
        log_prob = dist.log_prob(action) # Calc log prob of action
        # Return the action, state value, and log prob, all moved back to the CPU and converted to numpy
        return action.cpu().numpy().squeeze(), value.cpu().item(), log_prob.cpu().item()

    # Update the policy based on the collected experience.
    def learn(self, states, actions, log_probs_old, rewards, masks, values, images):  
        # To GPU
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        images = torch.tensor(np.array(images), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        masks = torch.tensor(np.array(masks), dtype=torch.float32).to(device)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(device)

        # Calc sum of future rewards for each state
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + 0.99 * R * masks[t]
            returns[t] = R
        # Normalize the returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - values # how much better was the return than expected?
        # Update Policy
        self.ppo.update(states, actions, images, log_probs_old, returns, advantages) 

# THis class manages everything around the simulations
class RunManager:
    def __init__(self, base_dir, run_name):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.run_dir = self.base_dir / run_name
        self.best_reward = -np.inf
        self.best_episode_dir = None
        self.best_episodes = []

    # Creates a directory if it does not exist
    def create_directory(self, directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"Unable to create directory {directory}. {e}")

    # Creates the directories for the run,
    # including script_info needed to load the run if ever stopped
    def create_directories(self, load_previous_run):
        self.create_directory(self.base_dir)
        if not load_previous_run and self.run_dir.exists():
            self.run_name = self.create_unique_run_name()
            self.run_dir = self.base_dir / self.run_name
        self.create_directory(self.run_dir)
        self.script_info_path = self.run_dir / 'script_info.txt'

    # Creates a unique run name if the given already exists
    def create_unique_run_name(self):
        i = 1
        while True:
            new_run_name = f"{self.run_name}_{i}"
            if not (self.base_dir / new_run_name).exists():
                return new_run_name
            i += 1

    # Updates the best episode if the reward of the current episode is better than the previous best.
    # It also handles copying the best episode files to the 'best_episode' directory.
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

    # Deletes the directories which should not be kept
    def maintain_last_episodes_memory(self, episode):
            episode_dirs = sorted(glob.glob(str(self.run_dir / 'Episode_*')), key=lambda x: int(x.split('_')[-1]))
            while len(episode_dirs) > LAST_EPISODES_MEMORY:
                if int(episode_dirs[0].split('_')[-1]) % SAVE_EVERY_X_EPISODE != 0:  # Skip the episodes that should be saved
                    print(f"Deleting directory: {episode_dirs[0]}") 
                    shutil.rmtree(episode_dirs[0], ignore_errors=True)
                episode_dirs.pop(0)

    # Loads the Weights & Biases ID for the current run
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

# Initialize the environment and the agent
env = ReachEnv()
agent = Agent()
# Creates directories
run_manager = RunManager(base_dir=BASE_DIR, run_name=RUN_NAME)
run_manager.create_directories(load_previous_run=LOAD_PREVIOUS_RUN) 
# initialieses div
states_history = [] 
best_reward = -np.inf
best_episode_dir = None
best_episodes = []
best_episode = -1


# If loading from a previous run, load the saved model state, optimizer state, and the best reward.
# Also load various saved data like states, actions, log probabilities, values, masks, and images.
# Otherwise, initialize the starting episode to 0.
if LOAD_PREVIOUS_RUN:
    save_info = torch.load(PREVIOUS_RUN_PATH)
    agent.policy.load_state_dict(save_info['model_state_dict']) 
    agent.optimizer.load_state_dict(save_info['optimizer_state_dict'])
    run_manager.best_reward = save_info['best_reward']
    start_episode = save_info['episode'] + 1
    best_episode_dir = os.path.dirname(PREVIOUS_RUN_PATH)

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

# Write script info to script_info
script_info = f'Script: {__file__}\n'
script_info += f'Script Path: {os.path.abspath(__file__)}\n'
script_info_path = run_manager.run_dir / 'script_info.txt'
with open(script_info_path, 'w') as f:
    f.write(script_info)

if USE_WANDB:
    import wandb
    wandb.login()

    # If continue a wandb run load the id from script info
    if WANDB_CONTINUE:
        wandb_id = run_manager.load_wandb_id(PREVIOUS_RUN_PATH)

    if WANDB_CONTINUE and wandb_id: 
        run = wandb.init(id=wandb_id,
                         resume="Force",
                         project=WANDB_PROJECT_NAME,
                         save_code=True)
    else: # If not, or if there is no previous ID, start a new run with a new ID.
        wandb_id = wandb.util.generate_id()
        run = wandb.init(project=WANDB_PROJECT_NAME,
                         id=wandb_id,
                         save_code=True,
                         config={
                             "learning_rate": LR,
                             "episodes": EPISODES,
                             "episode_length": EPISODE_LENGTH
                         })
    if run: #write wandb_id to script_info
        wandb_id = run.id
        script_info += f'Wandb ID: {wandb_id}\n'  
        with open(run_manager.script_info_path, 'w') as f:  
            f.write(script_info)

#Start loop for episodes 
for e in range(start_episode, EPISODES):
    print('Starting episode %d' % e)
    # Resets the env, and initializes div
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

    # Crete dir for current episode
    episode_dir = os.path.join(run_manager.run_dir, f'Episode_{e}') 
    os.makedirs(episode_dir, exist_ok=True)

    # Loop for every step within a episode
    for i in range(EPISODE_LENGTH):
        # Agent decides action based on state and image
        action, value, log_prob = agent.act(state, image) 
        # Append stores stuff
        states.append(state)
        images.append(image)
        actions.append(action)
        log_probs_old.append(log_prob)
        values.append(value)
        # The Action is executed, this gives a reward and the next state
        reward, (next_state, next_image), done = env.step(action, e, best_episode if best_episode >= 0 else e)        
        rewards.append(reward)
        masks.append(not done)
        # Update new state and image
        state = next_state
        image = next_image
        depth_images.append(image[:, :, 3])
        actions_history.append(action)
        # Calc the current distance to target and accumulate it
        distance_to_target = np.linalg.norm(env.agent_ee_tip.get_position() - env.waypoint.get_position())
        total_distance += distance_to_target
        last_distance = distance_to_target
    action_log_probs = np.array(action_log_probs)
    
    # After the steps and the episode is done it's time for the agent to learn
    agent.learn(states, actions, log_probs_old, rewards, masks, values, images)
    
    # Calculate things to store and track
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
    # Saves it all in model.pth so it can be reloaded
    torch.save(save_info, os.path.join(episode_dir, 'model.pth'))

    # Stores div for later analysis
    np.save(os.path.join(episode_dir, 'states.npy'), states)
    np.save(os.path.join(episode_dir, 'actions.npy'), actions)
    np.save(os.path.join(episode_dir, 'log_probs_old.npy'), log_probs_old)
    np.save(os.path.join(episode_dir, 'values.npy'), values)
    np.save(os.path.join(episode_dir, 'masks.npy'), masks)
    np.save(os.path.join(episode_dir, 'images.npy'), images)
    np.save(os.path.join(episode_dir, 'depth_images.npy'), depth_images)

    # Writes the name and of this script in script_info.txt
    shutil.copy(os.path.join(run_manager.run_dir, 'script_info.txt'), episode_dir)
    
    # Updates best episode if this is the highest rewarded episode in the run
    best_episode_dir = run_manager.update_best_episode(episode_reward, e, episode_dir, agent)
    
    # Deletes oldest directory if it should
    run_manager.maintain_last_episodes_memory(e)

# Tried OS alternative
BASE_DIR = Path(BASE_DIR)
RUN_NAME = Path(RUN_NAME)
BEST_EPISODE_DIR = Path(best_episode_dir)

# Creates RGB video and Depth video of the best run
make_video_command = join(dirname(abspath(__file__)), 'commands', 'make_video.py')
subprocess.run([make_video_command, f'{BEST_EPISODE_DIR}/images.npy'])
subprocess.run([make_video_command, f'{BEST_EPISODE_DIR}/depth_images.npy'])

print('Done!')
env.shutdown()

if USE_WANDB:
    wandb.finish()