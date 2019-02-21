import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_test_figure(frame, green_line_coords, blue_line_coords) :

    # Generating figure 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(frame)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(frame)

    ax[1].plot((green_line_coords[0],green_line_coords[2]), (green_line_coords[1], green_line_coords[3]), '-w')
    ax[1].scatter(x=green_line_coords[0], y=green_line_coords[1], marker='o', c='r', edgecolor='b')
    ax[1].scatter(x=green_line_coords[2], y=green_line_coords[3], marker='o', c='r', edgecolor='b')
    ax[1].plot((blue_line_coords[0], blue_line_coords[2]), (blue_line_coords[1], blue_line_coords[3]), '-w')
    ax[1].scatter(x=blue_line_coords[0], y=blue_line_coords[1], marker='o', c='r', edgecolor='b')
    ax[1].scatter(x=blue_line_coords[2], y=blue_line_coords[3], marker='o', c='r', edgecolor='b')

    ax[1].set_xlim((0, frame.shape[1]))
    ax[1].set_ylim((frame.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()