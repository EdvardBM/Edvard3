#!/usr/bin/env python3
import os
import argparse

def resume_training(run_dir, create_new_directory=False):
    # Get the latest episode directory
    episode_dirs = sorted([d for d in os.listdir(run_dir) if d.startswith('Episode_')], reverse=True)
    if not episode_dirs or create_new_directory:
        episode_number = 0
    else:
        latest_episode_dir = os.path.join(run_dir, episode_dirs[0])
        episode_number = int(episode_dirs[0].split('_')[-1])

    episode_number += 1
    episode_dir = os.path.join(run_dir, f"Episode_{episode_number}")
    os.makedirs(episode_dir, exist_ok=True)

    if create_new_directory:
        print(f"Starting a new episode {episode_number}")
    else:
        print(f"Continuing from episode {episode_number}")

    # Load necessary parameters and script information
    model_path = os.path.join(episode_dir, "model.pth")
    states_path = os.path.join(episode_dir, "states.npy")
    actions_path = os.path.join(episode_dir, "actions.npy")
    log_probs_path = os.path.join(episode_dir, "log_probs_old.npy")
    values_path = os.path.join(episode_dir, "values.npy")
    masks_path = os.path.join(episode_dir, "masks.npy")
    images_path = os.path.join(episode_dir, "images.npy")
    script_info_path = os.path.join(run_dir, "script_info.txt")

    # Load and use the necessary parameters and script information for training
    # ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume training for an agent")
    parser.add_argument("run_dir", type=str, help="Path to the run directory")
    parser.add_argument("--create-new-directory", action="store_true", help="Create a new episode directory")
    args = parser.parse_args()

    resume_training(args.run_dir, args.create_new_directory)
