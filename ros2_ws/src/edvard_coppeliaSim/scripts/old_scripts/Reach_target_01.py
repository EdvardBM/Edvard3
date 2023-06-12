#!/usr/bin/env python3

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.objects.dummy import Dummy

import numpy as np

SCENE_FILE = join(dirname(abspath(__file__)), 
                  '../../scenes/CSROS2_1reach_target.ttt')

# Waypoint pos min / max
POS_MIN, POS_MAX = [-1.7500e+00, +1.2500e-01, +7.0200e-01], [-1.2750e+00, +8.5000e-01, +7.0200e-01]
EPISODES = 5
EPISODE_LENGTH = 200


class ReachEnv(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.waypoint = Dummy('Waypoint')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    # Define the minimum and maximum positions for the waypoint
    WAYPOINT_POS_MIN, WAYPOINT_POS_MAX = [-1.7250e+00, +7.0000e-01, +7.2500e-01], [-1.3250e+00, +7.0000e-01, +7.2500e-01]
    
    def _get_state(self):
        # Return state containing arm joint angles/velocities & position of the waypoint
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.waypoint.get_position()])

    def reset(self):
        # Get a random position within a cuboid and set the waypoint position
        pos = list(np.random.uniform(self.WAYPOINT_POS_MIN, self.WAYPOINT_POS_MAX))
        self.waypoint.set_position(pos)

        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()
        
    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
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

    def act(self, state):
        del state
        return list(np.random.uniform(-1, 1, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass

env = ReachEnv()
agent = Agent()
replay_buffer = []

for e in range(EPISODES):

    print('Starting episode %d' % e)
    state = env.reset()
    for i in range(EPISODE_LENGTH):
        action = agent.act(state)
        reward, next_state, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state
        agent.learn(replay_buffer)

print('Done!')
env.shutdown()
