# coding=utf-8
__author__    = 'Dušan Marjanski <marjanskid@yahoo.com>'
__date__      = '31 January 2019'
__copyright__ = 'Copyright (c) 2019 Dušan Marjanski'

import cv2
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage.transform import (hough_line, hough_line_peaks)

import tensorflow as tf

# my imports
import neural_network as nn
import digit_tracker as dt

number_size = 28
number_img_rows = 28
number_img_cols = 28

import time

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
    min_boundary = np.array([200, 200, 200])

    only_numbers_image = cv2.inRange(frame_copy, min_boundary, max_boundary)
    #cv2.imshow("number_image", only_numbers_image)
    #only_numbers_image = cv2.medianBlur(only_numbers_image,3)
    #cv2.imshow("medianBlur", only_numbers_image)

    # ovde ima prostora za štimanje kad dođe do procenata
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    iterations = 2
    number = cv2.dilate(only_numbers_image, kernel, iterations)
    #cv2.imshow("dilate", number)
    number = cv2.erode(number, kernel, iterations=1)
    #cv2.imshow("erode", number)

    ###prikaz brojeva###
    #frame_copy = cv2.bitwise_and(frame_copy, frame_copy, mask=number)
    #cv2.imshow('Numbers', frame_copy)

    # provera### - postavljanje belog okvira oko cifre, dok su cifre crne boje
    #blur = cv2.GaussianBlur(only_numbers_image, (5, 5), 0)
    #thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    #cv2.imshow("thresh", thresh)

    #number = only_numbers_image
    #cv2.imshow("dilate + erode", number)

    return cv2.findContours(number, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 4 funkcija koja podesava granice regiona cifre
def shift_number_bounds(x, y, h, w):
    y_shift = int((number_img_rows - h)/2)
    x_shift = int((number_img_cols - w)/2)
    if (y - y_shift) >= 0:
        y -= y_shift
    else:
        y = 0

    if (x - x_shift) >= 0:
        x -= x_shift
    else:
        x = 0

    x_bound = x + number_img_cols
    if x_bound > frame_width:
        x_bound = frame_width

    y_bound = y + number_img_rows
    if y_bound > frame_height:
        y_bound = frame_height

    return x, y, x_bound, y_bound

# 5 funkcija koja proverava da li su visina i sirina regiona broja odgovarajuce
def check_number_boundaries(number_region):
    number_height = number_region.shape[0]
    if number_height < number_img_rows:
        return False

    number_width = number_region.shape[1]
    if number_width < number_img_cols:
        return False

    return True

################### Glavni tok programa ###################
#
#
# Kreiranje zaglavlja u fajlu za ispis dobijenih rezultata
output_file = open('out.txt', 'w')
output_file.write("RA31/2015 Dusan Marjanski" + '\n' + "file" + '\t' + "sum" + '\n')

for i in range(0,10):
    # Kreiranje VideoCapture objekta na osnovu prosledjene putanje do fajla
    video_path = "video-" + str(i) + ".avi"
    cap = cv2.VideoCapture(video_path)

    # Provera da li je camera uspesno pokrenuta
    if (cap.isOpened() == False):
        print("Greska pri otvaranju video snimka sa prosledjene putanje : " + video_path)

    frame_number = 0
    ret, detect_line_image = cap.read()

    blue_line_coords = find_line(detect_line_image, "blue")
    green_line_coords = find_line(detect_line_image, "green")

    #print("blue: " + str(blue_line_coords))
    #print("green: " + str(green_line_coords))

    """ Inicijalizacija tracker-a za brojeve """
    digit_tracker = dt.DigitTracker(blue_line_coords, green_line_coords)

    frame_height = detect_line_image.shape[0]
    frame_width = detect_line_image.shape[1]

    # Citaj do zavrsetka snimka
    while cap.isOpened():
        # Snimaj frejm po frejm
        ret, frame = cap.read()
        frame_number += 1
        check_frame = frame_number % 5
        if ret is True:
            #if check_frame is 1:
            # Prikaz svakog frejma
            #cv2.imshow(video_path, frame)
            # print(frame_number)

            # Pretvaranje frejma u grayscale
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow(video_path, gray_scale)

            """ Test funkcije """
            #generate_test_figure(frame, green_line_coords, blue_line_coords)
            """ Test funkcije """

            img, contours, hierarchy = get_number_contours(frame)
            prediction_contours_found = []
            tracking_contours_found = []
            frame_copy = frame.copy()
            #print("########## novi frejm no: " + str(frame_number) + " ###############")
            for i in range(len(contours)):
                number_contour = contours[i]
                x, y, w, h = cv2.boundingRect(number_contour)

                if w < 6 or h < 11:
                    continue

                #if (h >= 15 and h <= 25) or (w > 10 and h >= 14) and (hierarchy[0][i][3] == -1):
                x, y, x_bound, y_bound = shift_number_bounds(x, y, h, w)
                number_region = frame_copy[y: y_bound, x: x_bound]
                #cv2.rectangle(frame, (x, y), (x_bound, y_bound), (200, 255, 0), 2)

                if check_number_boundaries(number_region) is False:
                    continue

                tracking_region = np.array([x, y, x_bound, y_bound])

                # dodavanje pronadjene konture u listu
                prediction_contours_found.append(number_region)
                tracking_contours_found.append(tracking_region.astype("int"))

                #--------- test ------------
                #cv2.imshow('number_region', number_region)
                #--------- test ------------
                cv2.rectangle(frame, (x, y), (x_bound, y_bound), (200, 255, 0), 2)
                
                #--------- test ------------
                #print('height: ' + str(number_region.shape[0]))
                #print('width: ' + str(number_region.shape[1]))
                #--------- test ------------

            # slanje digit_tracker-u svih pronadjenih kontura
            #if (frame_number is not 27):
            digit_tracker.update_digits(prediction_contours_found, tracking_contours_found)
            #print("trenutna suma: " + str(digit_tracker.sum))
            #digit_tracker.all_digits_to_str()
            # ispis trenutnog frejma sa okvirima oko kontura 
            #cv2.imshow('Rectangle number', frame)

            # Pritisak na taster Q za izlazak
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                    
        # Izlazak iz petlje
        else:
            break

    #print('stigli smo do kraja video')

    # Na kraju, osloboditi video objekat
    cap.release()
    # Zatvoriti sve prozore
    cv2.destroyAllWindows()

    print(video_path + '\t' + str(digit_tracker.sum) + '\n')

    output_file.write(video_path + '\t' + str(digit_tracker.sum) + '\n')

output_file.close()
