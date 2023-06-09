'''
This script, joint_state_republisher_to_robot.py, enables control of a physical robot using a GUI. 
It subscribes to the 'joint_states' topic, which receives joint positions from a GUI. 
Upon receiving a JointState message, it extracts joint positions, repackages them into a JointPosition message, 
and publishes it to the '/command' topic, which the robot subscribes to. 
Thus, it acts as an intermediary, translating joint state data from the GUI into commands for the robot.

Command Line Usage:
python3 joint_state_republisher_to_robot.py
'''


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sunrisedds_interfaces.msg import JointPosition, JointQuantity

class JointStateToRobotRepublisher(Node):
    def __init__(self):
        super().__init__('joint_state_to_robot_republisher')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)
        self.subscription
        self.publisher_ = self.create_publisher(JointPosition, '/command', 10)

    def joint_state_callback(self, joint_state_msg):
        joint_position_msg = JointPosition()
        joint_position_msg.header = joint_state_msg.header
        joint_position_msg.position = JointQuantity(
            a1=joint_state_msg.position[0],
            a2=joint_state_msg.position[1],
            a3=joint_state_msg.position[2],
            a4=joint_state_msg.position[3],
            a5=joint_state_msg.position[4],
            a6=joint_state_msg.position[5],
            a7=joint_state_msg.position[6],
        )
        self.publisher_.publish(joint_position_msg)

def main(args=None):
    rclpy.init(args=args)

    joint_state_to_robot_republisher = JointStateToRobotRepublisher()

    rclpy.spin(joint_state_to_robot_republisher)

    joint_state_to_robot_republisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
