#!/bin/bash

# Set the runtime directory 
export XDG_RUNTIME_DIR=/my_runtime_dir

# Update LD_LIBRARY_PATH, first is only relevant in edvard6.sif
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "/opt/CoppeliaSim" | paste -sd: -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/

# Set COPPELIASIM_ROOT
export COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/

# Set QT_QPA_PLATFORM_PLUGIN_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/

echo "Environment setup completed."

#How to use:
#chmod +x idun_setup.sh
#source ./idun_setup.sh