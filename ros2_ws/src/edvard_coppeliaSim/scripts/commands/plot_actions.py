#!/usr/bin/env python3

"""
This script reads an .npy file containing a sequence of joint actions and plots each joint's actions over time.

It first loads the action data from the input .npy file. The action data should be a 2D array where each row corresponds to a timestep and each column corresponds to a joint.

A new figure is created, and the actions for each joint are plotted as a time series. Each joint action series is labeled accordingly ('Joint 1', 'Joint 2', etc.) in the plot.

The plot is then saved as a .png file in the same directory as the input .npy file. The name of the output file is the same as the input file, but with the .npy extension replaced with .png.

Command line usage:
plot_actions <actions.npy>

Arguments:
actions.npy: Path to the numpy file containing the joint actions.

Example:
plot_actions actions.npy

Note: This script must be executable and located in a directory listed in the system's PATH to use the simplified 'plot_actions' command. Otherwise, it can be run with 'python3 plot_actions.py'.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_actions(file_path):
    # Load the action data
    actions = np.load(file_path)

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Plot the actions for each joint
    for i in range(actions.shape[1]):
        plt.plot(actions[:, i], label=f'Joint {i + 1}')

    # Add labels and a legend
    plt.xlabel('Timestep')
    plt.ylabel('Action')
    plt.legend()

    # Save the plot
    plt.savefig(file_path.replace('.npy', '.png'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <actions.npy>")
        sys.exit(1)

    plot_actions(sys.argv[1])
