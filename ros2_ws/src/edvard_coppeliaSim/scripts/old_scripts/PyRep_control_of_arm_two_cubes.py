#!/usr/bin/env python3

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.objects.shape import Shape
import numpy as np

SCENE_FILE = join(dirname(abspath(__file__)), 
                  '../../scenes/CSROS2_10_RL_2_cubes.ttt')

POS_MIN, POS_MAX = [-1.7250e+00, +7.0000e-01, +7.0200e-01], [-1.3250e+00, +7.0000e-01, +7.0200e-01]
EPISODES = 5
EPISODE_LENGTH = 200

# Workspace coordinates for cubes
[x_min, y_min, z_min] = [-1.7000e+00, +1.5000e-01, +7.2500e-01]
[x_max, y_max] = [-1.3250e+00, +6.0000e-01]

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

    def act(self, state):
        del state
        return list(np.random.uniform(-1, 1, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass


env = PusherEnv()
agent = Agent()
replay_buffer = []

for e in range(EPISODES):

    print('Starting episode %d' % e)
    state = env.reset()
    for i in range(EPISODE_LENGTH):
        action = agent.act(state)
        reward, next_state = env.step(action)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state
        agent.learn(replay_buffer)

print('Done!')
env.shutdown()
