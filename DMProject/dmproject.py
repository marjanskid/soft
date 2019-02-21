import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import (hough_line, hough_line_peaks)


# funkcija za pronalazenje odabrane linije
import find_line as fl
# skripta koja sadrzi funkcije za testiranje
import test_script as ts



# Kreiranje VideoCapture objekta na osnovu prosledjene putanje do fajla
video_path = "video-0.avi"
cap = cv2.VideoCapture(video_path)

# Provera da li je camera uspesno pokrenuta
if (cap.isOpened() == False):
    print("Greska pri otvaranju video snimka sa prosledjene putanje : " + video_path)

frame_number = 0
ret, detect_line_image = cap.read()
green_line_coords = fl.find_line(detect_line_image, "green")
blue_line_coords = fl.find_line(detect_line_image, "blue")


# Citaj do zavrsetka snimka
while cap.isOpened():
    # Snimaj frejm po frejm
    ret, frame = cap.read()
    frame_number += 1
    if ret == True:
        # Prikaz svakog frejma
        cv2.imshow(video_path, frame)
        #print(frame_number)
        # Pretvaranje frejma u grayscale
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow(video_path, gray_scale)
        
        # Classic straight-line Hough transform
        h, theta, d = hough_line(gray_scale)

        #ts.generate_test_figure(frame, green_line_coords, blue_line_coords)


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
