import cv2
import scipy
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import (hough_line, hough_line_peaks)

################### Funkcije ###################
# 1 funkcija za pronalazenje odabrane linije (zelena/crvena)
def find_line(image, color):

    colorless_img = image.copy()

    if color == "green":
        # na ovaj nacin zanemarujem plavi spektar boje
        colorless_img[:, :, 0] = 0
    else:
        # a ovde zeleni
        colorless_img[:, :, 1] = 0

    gray_scale_img = cv2.cvtColor(colorless_img, cv2.COLOR_BGR2GRAY)

    _, t = cv2.threshold(gray_scale_img, 25, 200, cv2.THRESH_BINARY)

    #img_edges = cv2.Canny(t, 25, 200, None, 3)

    min_line_length = 100
    max_line_gap = 10
    tresh = 100
    line = cv2.HoughLinesP(t, 1, np.pi / 180, tresh, None,
                           min_line_length, max_line_gap)
    #line = cv2.HoughLinesP(img_edges, 1, np.pi / 180, tresh, None, min_line_length, max_line_gap)

    if line is None:
        print("None je mater mu bezobraznu")
    a1 = min(line[:, 0, 0])
    b1 = max(line[:, 0, 1])
    a2 = max(line[:, 0, 2])
    b2 = min(line[:, 0, 3])

    return a1, b1, a2, b2

# 2 funkcija koja sluzi kao provera pronalazenja linija za sabiranje i oduzimanje
def generate_test_figure(frame, green_line_coords, blue_line_coords):

    # Generating figure 1
    _, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(frame)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(frame)

    ax[1].plot((green_line_coords[0], green_line_coords[2]),
               (green_line_coords[1], green_line_coords[3]), '-w')
    ax[1].scatter(x=green_line_coords[0], y=green_line_coords[1],
                  marker='o', c='b', edgecolor='w')
    ax[1].scatter(x=green_line_coords[2], y=green_line_coords[3],
                  marker='o', c='b', edgecolor='w')
    ax[1].plot((blue_line_coords[0], blue_line_coords[2]),
               (blue_line_coords[1], blue_line_coords[3]), '-w')
    ax[1].scatter(x=blue_line_coords[0], y=blue_line_coords[1],
                  marker='o', c='b', edgecolor='w')
    ax[1].scatter(x=blue_line_coords[2], y=blue_line_coords[3],
                  marker='o', c='b', edgecolor='w')

    ax[1].set_xlim((0, frame.shape[1]))
    ax[1].set_ylim((frame.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()

# 3 funkcija koja sluzi za izdvajanje brojeva
def get_number_contours(frame):
    frame_copy = frame.copy()
    max_boundary = np.array([255, 255, 255])
    min_boundary = np.array([150, 150, 150])

    only_numbers_image = cv2.inRange(frame_copy, min_boundary, max_boundary)

    #cv2.imshow("number_image", only_numbers_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    iterations = 2
    number = cv2.dilate(only_numbers_image, kernel, iterations)
    #cv2.imshow("dilate", number)
    frame_copy = cv2.bitwise_and(frame_copy, frame_copy, mask=number)

    #cv2.imshow('Numbers', frame_copy)

    #provera - postavljanje belog okvira oko cifre, dok su cifre crne boje
    #blur = cv2.GaussianBlur(only_numbers_image, (5, 5), 0)
    #thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    #cv2.imshow("thresh", thresh)

    return cv2.findContours(number, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

################### Glavni tok programa ###################
#
#
# Kreiranje VideoCapture objekta na osnovu prosledjene putanje do fajla
video_path = "video-9.avi"
cap = cv2.VideoCapture(video_path)

# Provera da li je camera uspesno pokrenuta
if (cap.isOpened() == False):
    print("Greska pri otvaranju video snimka sa prosledjene putanje : " + video_path)

frame_number = 0
ret, detect_line_image = cap.read()
green_line_coords = find_line(detect_line_image, "green")
blue_line_coords = find_line(detect_line_image, "blue")

# Citaj do zavrsetka snimka
while cap.isOpened():
    # Snimaj frejm po frejm
    ret, frame = cap.read()
    frame_number += 1
    if ret == True:
        # Prikaz svakog frejma
        cv2.imshow(video_path, frame)
        print(frame_number)
        # Pretvaranje frejma u grayscale
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow(video_path, gray_scale)
        
        # Classic straight-line Hough transform
        h, theta, d = hough_line(gray_scale)

        #generate_test_figure(frame, green_line_coords, blue_line_coords)
        img, contours, hierarchy = get_number_contours(frame)

        for i in range(0, len(contours)):

            number_contour = contours[i]
            x, y, w, h = cv2.boundingRect(number_contour)
            cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2)

            #rect = cv2.minAreaRect(number_contour)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        cv2.imshow("Rectangle number", frame)

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
