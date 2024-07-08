"""
This script is for an image processing task where SIFT (Scale-Invariant Feature Transform) is used to detect
and match features between pairs of images. The script reads pairs of images, detects keypoints,
computes descriptors, matches them using BFMatcher, and then draws and saves the matches.

Author: Reza Toosi
"""

import cv2
import os
import matplotlib
import matplotlib.pyplot as plt

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')


def run(name):
    """
    Function to detect and draw keypoint matches between two images.
    Saves the output images with detected matches.

    Parameters:
    - name: The base name of the image files to be processed.
    """
    # Read images in grayscale mode
    img1 = cv2.imread(f'{input_path}\\Single_{name}.png', 0)
    img2 = cv2.imread(f'{input_path}\\Full_{name}.png', 0)

    # Detect keypoints and compute descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    matches1 = bf.match(des1, des2)

    # Sort matches based on distance
    matches1 = sorted(matches1, key=lambda val: val.distance)

    # Draw the first 25 matches with different flags
    out0 = cv2.drawMatches(img1, kp1, img2, kp2, matches1[:25], None, flags=0)
    out1 = cv2.drawMatches(img1, kp1, img2, kp2, matches1[:25], None, flags=2)

    # Save the output images
    plt.imshow(out0)
    plt.savefig(f'{result_path}\\{name}1.png', dpi=300)

    plt.imshow(out1)
    plt.savefig(f'{result_path}\\{name}2.png', dpi=300)


# Initialize SIFT detector
sift = cv2.SIFT_create()

# Create BFMatcher object
bf = cv2.BFMatcher()

# Create a directory to save the results if it does not exist
os.makedirs('./Result', exist_ok=True)
result_path = f'{os.getcwd()}\\Result'

# Specify the input directory
input_path = f'{os.getcwd()}\\Input'

# Run the function for different image sets
run('Shape')
run('Logo')
run('People')
