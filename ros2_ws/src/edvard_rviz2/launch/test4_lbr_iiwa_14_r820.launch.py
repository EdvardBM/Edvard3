'''
Here the need to write use_joint_state_republisher:=False when using the gui is removed.
Now the command is:
ros2 launch kuka_lbr_iiwa_support test4_lbr_iiwa_14_r820.launch.py use_joint_state_gui:=True
'''
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

class CustomJointStateRepublisher(LaunchConfiguration):
    def __init__(self, use_joint_state_gui: str):
        super().__init__(use_joint_state_gui)
        self.__use_joint_state_gui = LaunchConfiguration(use_joint_state_gui)

    def perform(self, context=None) -> str:
        return 'True' if self.__use_joint_state_gui.perform(context) == 'False' else 'False'


def generate_launch_description():

    # Get the package directory
    kuka_lbr_iiwa_support_package_dir = get_package_share_directory('kuka_lbr_iiwa_support')

    # Define default paths for RVIZ config and URDF model
    default_rviz_config_path = os.path.join(kuka_lbr_iiwa_support_package_dir, 'rviz/rviz_basic_settings.rviz')
    default_urdf_model_path = os.path.join(kuka_lbr_iiwa_support_package_dir, 'urdf/lbr_iiwa_14_r820.urdf')

    # Define launch configurations
    use_joint_state_republisher = CustomJointStateRepublisher('use_joint_state_gui')
    urdf_model = LaunchConfiguration('urdf_model')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_joint_state_gui = LaunchConfiguration('use_joint_state_gui')

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

    declare_use_joint_state_gui_cmd = DeclareLaunchArgument(
        name='use_joint_state_gui',
        default_value='False',
        description='Whether to start the joint state publisher GUI')

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

    start_joint_state_publisher_gui_cmd = Node(
        condition=IfCondition(use_joint_state_gui),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen')

    ld = LaunchDescription()

    ld.add_action(declare_urdf_model_path_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_use_robot_state_pub_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_joint_state_republisher_cmd)
    ld.add_action(declare_use_joint_state_gui_cmd)

    ld.add_action(start_joint_state_republisher_cmd)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_rviz_cmd)
    ld.add_action(start_joint_state_publisher_gui_cmd)

    return ld


