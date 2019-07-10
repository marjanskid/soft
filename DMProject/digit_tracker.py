# coding=utf-8
__author__    = 'Dušan Marjanski <marjanskid@yahoo.com>'
__date__      = '31 May 2019'
__copyright__ = 'Copyright (c) 2019 Dušan Marjanski'

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import time
import operator
import math

import matplotlib.pyplot as plt

# my imports
import neural_network as nn

number_size = 28
number_img_rows = 28
number_img_cols = 28

""" Rastojanje izmedju dve tacke u prostoru """
def distance_between_two_spots(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

""" Funkcija koja vraca da li je distanaca izmedju ivice linije i tacke manja od prosledjene vrednosti """
def distance_from_line_less_than(line, A, distance):
    if distance_between_two_spots(line[0], line[1], A.get_x(), A.get_y()) < distance:
        return True

    if distance_between_two_spots(line[2], line[3], A.get_x(), A.get_y()) < distance:
        return True

    return False

class DigitTracker:
    def __init__(self, blue_line, green_line, max_disappeared=50):
        self.next_digit_id = 0
        self.digits_spoted = OrderedDict()
        self.digits_disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given
		# digit is allowed to be marked as "disappeared" until we
		# need to deregister the digit from tracking
        self.max_disappeared = max_disappeared
        self.sum = 0
        self.blue_line = (blue_line[0], blue_line[1] , blue_line[2] , blue_line[3])
        self.green_line = (green_line[0] , green_line[1] , green_line[2] , green_line[3])
        self.neural_network = nn.NeuralNetwork()

    def register_digit(self, digit):
        # when registering an digit we use the next available digit
		# ID to store the centroid
	    self.digits_spoted[self.next_digit_id] = digit
	    self.digits_disappeared[self.next_digit_id] = 0
	    self.next_digit_id += 1

    def deregister_digit(self, digit_id):
        self.include_digit_in_sum(digit_id)
        # to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
        del self.digits_spoted[digit_id]
        del self.digits_disappeared[digit_id]

    def update_digits(self, prediction_contours_found, tracking_contours_found):
        if len(prediction_contours_found) == 0:
            # loop over any existing tracked digits and mark them
		    # as disappeared
            for digit_id in self.digits_disappeared.keys():
                self.digits_disappeared[digit_id] += 1
        
        # initialize an array of Digits for the current frame
        input_digits = []
		# loop over the bounding box rectangles
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(tracking_contours_found):
			# use the bounding box coordinates to derive the centroid
            center_x = int((start_x + end_x) / 2.0)
            center_y = int((start_y + end_y) / 2.0)
            # predict the number using trained neural network
            predicted_number = self.neural_network.predict_number(prediction_contours_found[i], number_img_rows, number_img_cols)
            #--------- deo za proveru broja ------------
            #print(str(predicted_number))
            #cv2.imshow(str(predicted_number), prediction_contours_found[i])
            #time.sleep(.5)
            #--------- deo za proveru broja ------------
            digit = Digit(DigitCoords(center_x, center_y))
            digit.update_digit_prediction(predicted_number)
            input_digits.append(digit)

        # if we are currently not tracking any digits take the input
		# centroids and register each of them
        if len(self.digits_spoted) == 0:
            #print("samo nove")
            for digit in input_digits:
                digit.set_id(self.next_digit_id)
                self.register_digit(digit)
        else:
            #print("e, jeba ga ti - sledi seks")
            # grab the set of object IDs and corresponding centroids
            digit_ids = list(self.digits_spoted.keys())
            current_digits = list(self.digits_spoted.values())
            #print("ucitani id-jevi i cifre postojecih")

            # priprema koordinata postojecih cifara
            current_digit_coords = self.get_coords_of_all_digits(current_digits)
            #print("uspeo sam i postojece koordinate da smestim u neki niz")

            # priprema koordinata cifara sa trenutnog frejma
            frame_digit_coords = self.get_coords_of_all_digits(input_digits)
            #print("uspeo sam i nove koordinate da smestim u neki niz")

            # compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
            Distance = dist.cdist(current_digit_coords, frame_digit_coords)
            rows = Distance.min(axis=1).argsort()
            cols = Distance.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
            used_rows = set()
            used_cols = set()

            # loop over the combination of the (row, column) index
			# tuples
            for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
                if row in used_rows or col in used_cols:
                    continue

                # otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
                digit_id = digit_ids[row]
                spoted_digit = input_digits[col]
                matching_digit = self.digits_spoted[digit_id]
                matching_digit.set_x_coord(spoted_digit.get_x_coord())
                matching_digit.set_y_coord(spoted_digit.get_y_coord())
                matching_digit.update_digit_prediction(spoted_digit.get_most_predicted_digit())

                #matching_digit = self.match
                self.digits_spoted[digit_id] = matching_digit
                self.digits_disappeared[digit_id] = 0

                # probaj da je ubacis u sumu ako je presla neku od linija
                self.include_digit_in_sum(digit_id)

                # indicate that we have examined each of the row and
				# column indexes, respectively
                used_rows.add(row)
                used_cols.add(col)

                # compute both the row and column index we have NOT yet
			    # examined
                unused_rows = set(range(0, Distance.shape[0])).difference(used_rows)
                unused_cols = set(range(0, Distance.shape[1])).difference(used_cols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if Distance.shape[0] >= Distance.shape[1]:
                # loop over the unused row indexes
                for row in unused_rows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    digit_id = digit_ids[row]
                    self.digits_disappeared[digit_id] += 1

                    # check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
                    if self.digits_disappeared[digit_id] > self.max_disappeared:
                        self.deregister_digit(digit_id)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                #print("treba dodati novu cifru")
                for col in unused_cols:
                    new_digit = input_digits[col]
                    new_digit.set_id(self.next_digit_id)
                    self.register_digit(new_digit)

    def include_digit_in_sum(self, digit_id):
        digit = self.digits_spoted[digit_id]
        digit.check_for_crossing_lines(self.blue_line, self.green_line)
        #print("............include_digit_in_sum................")
        #print("plava predjena: " + str(digit.is_blue_line_crossed()))
        #print("zelena predjena: " + str(digit.is_green_line_crossed()))
        if digit.blue_counted is False:
            if digit.is_blue_line_crossed() > 5 or distance_from_line_less_than(self.blue_line, digit.digit_coords, 5):
                #print("plava: " + str(digit.get_most_predicted_digit()))
                self.digits_spoted[digit_id].blue_counted = True
                self.sum += digit.get_most_predicted_digit()
        
        if digit.green_counted is False:
            if digit.is_green_line_crossed() > 5 or distance_from_line_less_than(self.green_line, digit.digit_coords, 5):
                #print("zelena: " + str(digit.get_most_predicted_digit()))
                self.digits_spoted[digit_id].green_counted = True
                self.sum -= digit.get_most_predicted_digit()   
 
    def get_coords_of_all_digits(self, digits):
        digit_coords = np.zeros((len(digits), 2), dtype="int")
        for i in range(0, len(digits)):
            x = digits[i].digit_coords.get_x()
            y = digits[i].digit_coords.get_y()
            digit_coords[i] = (x, y)

        return digit_coords

    def all_digits_to_str(self):
        #print("Current spoted digits: ")
        for d in list(self.digits_spoted.values()):
            print("id: " + str(d.get_id()) + ", coords: [" 
                + str(d.get_x_coord()) + " : " + str(d.get_y_coord()) 
                + "], number: " + str(d.get_most_predicted_digit()))

class Digit:
    def __init__(self, digit_coords):
        self.blue_line_crossed = 0
        self.blue_counted = False
        self.green_line_crossed = 0
        self.green_counted = False
        self.first_time_coords = DigitCoords(digit_coords.get_x(), digit_coords.get_y())
        self.digit_coords = DigitCoords(digit_coords.get_x(), digit_coords.get_y())
        self.digit_prediction = OrderedDict()
        for digit in range(0, 10):
            self.digit_prediction[digit] = 0  

    def set_id(self, digit_id):
        self.id = digit_id

    def get_id(self):
        return self.id

    def is_blue_line_crossed(self):
        return self.blue_line_crossed

    def set_blue_line_crossed(self, value):
        self.blue_line_crossed = value

    def is_green_line_crossed(self):
        return self.green_line_crossed

    def set_green_line_crossed(self, value):
        self.green_line_crossed = value

    def get_digit_prediction(self):
        self.digit_prediction_to_str()
        return self.digit_prediction

    def update_digit_prediction(self, predicted_number):
        self.digit_prediction[predicted_number] += 1

    def set_x_coord(self, new_center_x):
        self.digit_coords.set_x(new_center_x)

    def get_x_coord(self):
        return self.digit_coords.get_x()

    def set_y_coord(self, new_center_y):
        self.digit_coords.set_y(new_center_y)
    
    def get_y_coord(self):
        return self.digit_coords.get_y()

    def get_first_digit_coords(self):
        return self.first_time_coords

    def get_most_predicted_digit(self):
        return max(self.digit_prediction.keys(), key=(lambda k: self.digit_prediction[k]))

    def ccw(self,A,B,C):
        return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

    def intersect(self,A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def check_for_crossing_lines(self, blue_line, green_line):
        """Proverava da li je presecena linija."""
        C = DigitCoords(self.first_time_coords.get_x(), self.first_time_coords.get_y())
        D = DigitCoords(self.digit_coords.get_x(), self.digit_coords.get_y())
        #print("first time coords: [ " + str(C.get_x()) + " : " + str(C.get_y()) + " ]")
        #print("last time coords: [ " + str(D.get_x()) + " : " + str(D.get_y()) + " ]")
        #print("pre provere preseka plave i zelene")
        if self.blue_line_crossed < 20:
            A = DigitCoords(blue_line[0], blue_line[1])
            B = DigitCoords(blue_line[2], blue_line[3])           
            if self.intersect(A,B,C,D):
                self.blue_line_crossed += 1

        if self.green_line_crossed < 20:
            A = DigitCoords(green_line[0], green_line[1])
            B = DigitCoords(green_line[2], green_line[3])            
            if self.intersect(A,B,C,D):
                self.green_line_crossed += 1

    def digit_prediction_to_str(self):
        for digit in range(0, 10):
            print("[" + str(digit) + "] = " + str(self.digit_prediction[digit]))


# ovo neka stoji tu implementirano - mozda se iskoristi, zlu ne trebalo
class DigitCoords:
    def __init__(self, center_x, center_y):
        self.x = center_x
        self.y = center_y

    def get_x(self):
        return self.x
    
    def set_x(self, new_center_x):
        self.x = new_center_x

    def get_y(self):
        return self.y

    def set_y(self, new_center_y):
        self.y = new_center_y

        

