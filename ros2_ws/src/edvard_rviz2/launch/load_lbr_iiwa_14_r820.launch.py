
import os
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    kuka_lbr_iiwa_support_package_dir = get_package_share_directory('kuka_lbr_iiwa_support')

    xacro_file = os.path.join(kuka_lbr_iiwa_support_package_dir, 'urdf', 'lbr_iiwa_14_r820.xacro')

    robot_description = {
        'robot_description': Command(['xacro', ' ', xacro_file])
    }

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[robot_description]
        )
    ])