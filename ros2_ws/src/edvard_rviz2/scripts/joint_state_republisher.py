#!/usr/bin/env python3

'''
This script defines a ROS2 node, joint_state_republisher, that subscribes to JointPosition messages, 
republishes the received data as JointState messages, and updates the current joint states of a robot.

This node subscribes to the state topic where it expects to receive JointPosition messages that carry the current status of the robot's joints. 
When a message is received, it triggers the robot_current_status callback, which updates the joint state values.

In addition to the subscription, the node also creates a publisher that publishes JointState messages to the /joint_states topic. 
It initializes an array of joint states and joint names to keep track of the robot's status.

To ensure continuous updates, the node creates a timer that triggers the publish_joint_states method at a rate of 10Hz. 
This method constructs a JointState message with the current joint states and publishes it.

Finally, the script's main function initializes and runs the node until an interruption is received or the script is terminated, at which point the node is destroyed and ROS2 is shut down.

Command Line Usage:
python3 joint_state_republisher.py
'''


# Import required libraries
import rclpy
from rclpy.node import Node
from sunrisedds_interfaces.msg import JointPosition
from sensor_msgs.msg import JointState
import numpy as np

# Define JointStateRepublisher class which inherits from Node
class JointStateRepublisher(Node):
    def __init__(self):
        super().__init__('joint_state_republisher')

        # Create a subscription to JointPosition messages
        self.subscription_robot = self.create_subscription(
            JointPosition,
            'state',
            self.robot_current_status,
            1)

        # Create a publisher for JointState messages
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            1)

        # Initialize joint states and joint names
        self.joint_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_names = ['joint_a1', 'joint_a2', 'joint_a3', 'joint_a4', 'joint_a5', 'joint_a6', 'joint_a7']

        # Create a timer to periodically publish joint states
        self.create_timer(
            0.1,  # Publish at 10Hz
            self.publish_joint_states)

    # Callback function to update joint states when a new JointPosition message is received
    def robot_current_status(self, msg):
        self.joint_state = [msg.position.a1, msg.position.a2, msg.position.a3, msg.position.a4, msg.position.a5, msg.position.a6, msg.position.a7]

    # Function to publish joint states as JointState messages
    def publish_joint_states(self):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self.joint_state

        self.joint_state_pub.publish(joint_state_msg)

# Main function to initialize and run the node
def main(args=None):
    rclpy.init(args=args)

    joint_state_republisher = JointStateRepublisher()

    rclpy.spin(joint_state_republisher)

    joint_state_republisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
