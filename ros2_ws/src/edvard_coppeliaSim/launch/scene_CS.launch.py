import os
import psutil
import subprocess

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def ask_for_scene():
    scene_dir = os.path.expanduser('~/Edvard2/ros2_ws/src/scenes/')
    print(f"Looking for scenes in the directory: {scene_dir}")
    scene_name = input("Please enter the name of the scene: ")

    # Add .ttt extension if not already present
    if not scene_name.endswith('.ttt'):
        scene_name += '.ttt'
    
    scene_path = os.path.join(scene_dir, scene_name)
    if not os.path.isfile(scene_path):
        create_new = input(f"The scene {scene_name} does not exist. Do you want to create it? (y/n): ")
        if create_new.lower() == 'y':
            with open(scene_path, 'w') as new_scene:
                print(f"Created a new scene at {scene_path}")
        else:
            print("Scene does not exist and will not be created.")
            scene_path = ""
    return scene_path

def check_process_and_start(context, *args, **kwargs):
    process_name = "coppeliaSim"
    coppeliasim_path = os.path.expanduser('~/CoppeliaSim_Edu/coppeliaSim.sh')

    # Get scene path from user input
    scene_path = ask_for_scene()

    if scene_path == "":
        return []

    # Check if process is already running
    for proc in psutil.process_iter(['pid', 'name']):
        if process_name.casefold() in proc.info['name'].casefold():
            print(f"{process_name} is already running.")
            return []
    
    # Start CoppeliaSim with the scene
    command = f'{coppeliasim_path} {scene_path}'
    return [ExecuteProcess(cmd=[command], shell=True)]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=check_process_and_start)
    ])
