#!/usr/bin/env python3

"""
This script converts a .npy file of either RGB or depth images into a video file. 

It first checks if the data in the numpy file represents RGB or Depth images. 
For Depth images, the script normalizes the data to fall between 0 and 1.

Each image from the numpy file is temporarily saved as a .png file in the 
same directory as the input file. 

A video is then created from the sequence of images using the moviepy library. 
The video is saved in a subdirectory named 'Video' in the same directory as the input file. 
The default names for the video files are 'RGB_video.mp4' or 'Depth_video.mp4', 
depending on the type of images. 

Finally, the temporary .png files are deleted to clean up the directory.

Command line usage:
    make_video <npy_path> [--fps <frames_per_second>] [--filename <output_filename>]

Arguments:
    npy_path: Path to the numpy file.
    fps: Frames per second for the output video. Default is 24.
    filename: Name of the output video file. Default is RGB_video.mp4 or Depth_video.mp4.

Example:
    make_video images.npy --fps 30 --filename my_video.mp4

Note: This script must be executable and located in a directory listed in the system's PATH 
to use the simplified 'make_video' command. Otherwise, it can be run with 'python3 make_video.py'.
"""

import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import glob
import os
import argparse

def make_video(npy_path, fps, filename):
    # Load the numpy file
    images = np.load(npy_path)

    # Check if the data is Depth or RGB
    if images.shape[-1] == 3 or images.shape[-1] == 4:
        image_type = 'RGB'
    else:
        image_type = 'Depth'

    # Process and save the images
    directory = os.path.dirname(npy_path) if os.path.dirname(npy_path) else '.'
    for idx, img in enumerate(images):
        if image_type == 'Depth':
            img = (img - np.min(img)) / (np.max(img) - np.min(img))  
        plt.imsave(f'{directory}/{idx:05d}.png', img)

    # Create the video
    image_files = sorted(glob.glob(f'{directory}/*.png'), key=lambda x: int(os.path.basename(x).split('.')[0]))
    clip = ImageSequenceClip(image_files, fps=fps)
    output_dir = os.path.join(directory, 'Video')
    os.makedirs(output_dir, exist_ok=True)
    
    # Default video file names
    if filename is None:
        filename = f'{image_type}_video.mp4'
    
    clip.write_videofile(f'{output_dir}/{filename}')

    # Clean up the image files
    for file in image_files:
        os.remove(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create video from npy file.')
    parser.add_argument('npy_path', type=str, help='Path to the numpy file.')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second for the output video. Default is 24.')
    parser.add_argument('--filename', type=str, default=None, help='Name of the output video file. Default is RGB_video.mp4 or Depth_video.mp4.')
    args = parser.parse_args()
    make_video(args.npy_path, args.fps, args.filename)
