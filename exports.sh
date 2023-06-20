#!/bin/bash

# Set the runtime directory 
export XDG_RUNTIME_DIR=/my_runtime_dir

# Update LD_LIBRARY_PATH, first is only relevant in edvard6.sif
# export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "/opt/CoppeliaSim" | paste -sd: -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/

# Set COPPELIASIM_ROOT
export COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/

# Set QT_QPA_PLATFORM_PLUGIN_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/

# Miniconda
export PATH=/opt/miniconda/bin:$PATH
export PATH=/root/.local/bin:$PATH

# Set up a virtual frame buffer, incrementing the display number until one is free
DISP_NUM=1300
while true; do
  if [ -e /tmp/.X${DISP_NUM}-lock ]; then
    # Display number in use, increment the DISP_NUM
    let DISP_NUM++
  else
    # Display number not in use, start Xvfb
    Xvfb :${DISP_NUM} -screen 0 1280x1024x24 &
    break
  fi
done

# Set your display
export DISPLAY=:${DISP_NUM}.0
echo $DISPLAY
echo "Environment setup completed."

# Use GPU
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/miniconda/lib/python3.10/site-packages/torch/lib



#How to use:
#chmod +x exports.sh
#source exports.sh