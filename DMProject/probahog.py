import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier  # KNN

from skimage import img_as_ubyte
from sklearn.datasets import fetch_mldata
from skimage.morphology import disk
from PIL import Image
from skimage.transform import (hough_line, hough_line_peaks)
from matplotlib import cm

# Kreiranje VideoCapture objekta na osnovu prosledjene putanje do fajla
video_path = "video-0.avi"
cap = cv2.VideoCapture(video_path)

# Provera da li je camera uspesno pokrenuta
if (cap.isOpened() == False):
    print("Greska pri otvaranju video snimka sa prosledjene putanje : " + video_path)

frame_number = 0
# Citaj do zavrsetka snimka
while cap.isOpened():
    # Snimaj frejm po frejm
    ret, frame = cap.read()
    frame_number += 1
    if ret == True:
        # Prikaz svakog frejma
        #cv2.imshow(video_path, frame)
        print(frame_number)
        # Pretvaranje frejma u grayscale
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow(video_path, gray_scale)
        
        # Classic straight-line Hough transform
        h, theta, d = hough_line(gray_scale)

        # Generating figure 1
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(frame)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(frame)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - frame.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[1].plot((0, frame.shape[1]), (y0, y1), '-w')
        ax[1].set_xlim((0, frame.shape[1]))
        ax[1].set_ylim((frame.shape[0], 0))
        ax[1].set_axis_off()
        ax[1].set_title('Detected lines')

        plt.tight_layout()
        plt.show()


        # Pritisak na taster Q za izlazak
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Izlazak iz petlje
    else:
        break

# Na kraju, osloboditi video objekat
cap.release()
# Zatvoriti sve prozore
cv2.destroyAllWindows()