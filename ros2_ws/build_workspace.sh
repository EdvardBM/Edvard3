#!/bin/bash

# Define your workspace location
ROS2_WS_PATH=~/Edvard3/ros2_ws

# Navigate to your workspace
cd $ROS2_WS_PATH

rm -r build install log

# Build each package individually
cd src/edvard_interfaces

rm -r build install log

colcon build --symlink-install

source install/setup.bash

cd ..

cd sunrisedds

rm -r build install log

colcon build --symlink-install

source install/setup.bash

# Build the entire workspace

cd $ROS2_WS_PATH
echo "Building entire workspace..."
ulimit -s unlimited
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --allow-overriding edvard_interfaces sunrisedds_interfaces
source install/setup.bash


#Before running 
#sudo chown -R $USER:$USER ~/Edvard3/ros2_ws
#bash build_workspace.sh 
