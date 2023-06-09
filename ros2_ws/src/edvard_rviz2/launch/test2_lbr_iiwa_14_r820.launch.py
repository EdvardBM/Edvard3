"""
This script is a launch file for the KUKA LBR IIWA robot, specifically the LBR IIWA 14 R820 model.
The launch file starts a joint state republisher, a robot state publisher, and RVIZ. 
The joint state republisher is responsible for converting custom robot joint messages into the standard JointState message format.
he robot state publisher reads the robot's URDF model and publishes the robot's state to TF.
Finally, RVIZ is started for visualizing the robot's state in a 3D environment.

The launch file imports necessary ROS packages, sets up default paths for the robot's URDF model and RVIZ configuration files, and defines launch configurations for controlling the behavior of the launched nodes. 
Launch arguments are declared to allow customization of the launch process. 
The launch file then defines node actions that will start the joint state republisher, robot state publisher, and RVIZ based on the specified launch configurations.

Files/scripts used in the launch file:
1. The robot's URDF model (lbr_iiwa_14_r820.urdf): Used by the robot state publisher for publishing robot state to TF.
2. RVIZ configuration files (rviz_basic_settings.rviz and robot_state_visualize.rviz): Used to customize the RVIZ visualization.
3. Joint state republisher script (joint_state_republisher.py): Responsible for converting custom robot joint messages into the standard JointState message format.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Get the package directory
    kuka_lbr_iiwa_support_package_dir = get_package_share_directory('kuka_lbr_iiwa_support')

    # Find the package share directory
    pkg_share = FindPackageShare(package='kuka_lbr_iiwa_support').find('kuka_lbr_iiwa_support')

    # Define default paths for RVIZ config and URDF model
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/rviz_basic_settings.rviz')
    default_urdf_model_path = os.path.join(pkg_share, 'urdf/lbr_iiwa_14_r820.urdf')

    # Define RVIZ config file path
    rviz_config_file = os.path.join(kuka_lbr_iiwa_support_package_dir, 'config', 'robot_state_visualize.rviz')
    
    # Define launch configurations
    use_joint_state_republisher = LaunchConfiguration('use_joint_state_republisher')
    urdf_model = LaunchConfiguration('urdf_model')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare launch arguments
    declare_urdf_model_path_cmd = DeclareLaunchArgument(
        name='urdf_model', 
        default_value=default_urdf_model_path, 
        description='Absolute path to robot urdf file')

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        name='rviz_config_file',
        default_value=default_rviz_config_path,
        description='Full path to the RVIZ config file to use')

    declare_use_joint_state_republisher_cmd = DeclareLaunchArgument(
        name='use_joint_state_republisher',
        default_value='True',
        description='Whether to start the joint state republisher')

    declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
        name='use_robot_state_pub',
        default_value='True',
        description='Whether to start the robot state publisher')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='True',
        description='Whether to start RVIZ')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true')

    # Define node actions
    start_joint_state_republisher_cmd = Node(
        condition=IfCondition(use_joint_state_republisher),
        package='kuka_lbr_iiwa_support',
        executable='joint_state_republisher.py',
        name='joint_state_republisher',
        output='screen',
        prefix='python3',)

    start_robot_state_publisher_cmd = Node(
        condition=IfCondition(use_robot_state_pub),
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time, 
        'robot_description': Command(['xacro ', urdf_model])}],
        arguments=[default_urdf_model_path])

    start_rviz_cmd = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file])


    ld = LaunchDescription()

    ld.add_action(declare_urdf_model_path_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_use_robot_state_pub_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_joint_state_republisher_cmd)

    ld.add_action(start_joint_state_republisher_cmd)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_rviz_cmd)

    return ld