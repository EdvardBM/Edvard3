#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sunrisedds_interfaces.msg import JointPosition, JointQuantity


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher_ = self.create_publisher(JointPosition, 'jointstatesfromgui', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = JointPosition()
        joint_data = JointQuantity()

       # Cast self.i to float
        joint_data.a1 = float(self.i)
        joint_data.a2 = float(self.i)
        joint_data.a3 = float(self.i)
        joint_data.a4 = float(self.i)
        joint_data.a5 = float(self.i)
        joint_data.a6 = float(self.i)
        joint_data.a7 = float(self.i)

        msg.position = joint_data

        self.i += 1
        self.publisher_.publish(msg)



def main(args=None):
    rclpy.init(args=args)

    joint_state_publisher = JointStatePublisher()

    rclpy.spin(joint_state_publisher)

    joint_state_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
