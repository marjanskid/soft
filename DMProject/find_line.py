import cv2
import numpy as np

def find_line(image, color) :

    colorless_img = image.copy()   

    if color == "green" :
        # na ovaj nacin zanemarujem plavu boju
        colorless_img[:, :, 0] = 0      
    else :
        # a ovde zelenu
        colorless_img[:, :, 1] = 0 

    gray_scale_img = cv2.cvtColor(colorless_img, cv2.COLOR_BGR2GRAY)

    ret, t = cv2.threshold(gray_scale_img, 25, 255, cv2.THRESH_BINARY)

    minLineLength = 100
    maxLineGap = 10
    tresh = 100
    line = cv2.HoughLinesP(t, 1, np.pi / 180, tresh, minLineLength, maxLineGap)

    x1 = min(line[:, 0, 0])
    y1 = max(line[:, 0, 1])
    x2 = max(line[:, 0, 2])
    y2 = min(line[:, 0, 3])

    return x1, y1, x2, y2