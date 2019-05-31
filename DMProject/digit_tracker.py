# coding=utf-8
__author__    = 'Dušan Marjanski <marjanskid@yahoo.com>'
__date__      = '31 May 2019'
__copyright__ = 'Copyright (c) 2019 Dušan Marjanski'

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class DigitTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.digits_spoted = OrderedDict()
        self.digits_disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
        self.max_disappeared = max_disappeared

    def register_digit(self, digit):
        # when registering an object we use the next available object
		# ID to store the centroid
	    self.digits_spoted[self.next_object_id] = digit
	    self.digits_disappeared[self.next_object_id] = 0
	    self.next_object_id += 1

    #def deregister_digit(self, digit_id)
        # razmisli da li ima smisla da se radi ovo
        # nije bitno sto je cifra nestala dokle god je
        # bar jedan od flegova za prelazak preko linije
        # za tu cifru True

    def update_digits(self, digits_in_frame):
        if len(digits_in_frame) == 0:
            # loop over any existing tracked objects and mark them
		    # as disappeared
            for digit_id in self.digits_disappeared.keys():
                self.digits_disappeared[digit_id] += 1

            return self.digits_spoted
        
        # initialize an array of input centroids for the current frame
        input_digits = np.zeros((len(digits_in_frame), 2), dtype="int")
 
		# loop over the bounding box rectangles
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(digits_in_frame):
			# use the bounding box coordinates to derive the centroid
            center_x = int((start_x + end_x) / 2.0)
            center_y = int((start_y + end_y) / 2.0)
            input_digits[i] = (center_x, center_y)

        # if we are currently not tracking any objects take the input
		# centroids and register each of them
        if len(self.digits_spoted) == 0:
            for i in range(0, len(input_digits)):
                digit = Digit(self.next_object_id, input_digits[i])
                self.register_digit(digit)
        else:
            print("e, jeba ga ti")



class Digit:
    def __init__(self, digit_id, digit_coords):
        self.id = digit_id
        self.blue_line_crossed = False
        self.green_line_crossed = False
        self.digit_coords = (digit_coords[0], digit_coords[1])
        self.digit_prediction = OrderedDict()
        for digit in range(0, 9):
            self.digit_prediction[digit] = 0  

    def get_digit_id(self):
        return self.id

    def get_blue_line_crossed(self):
        return self.blue_line_crossed

    def set_blue_line_crossed(self, value):
        self.blue_line_crossed = value

    def get_green_line_crossed(self):
        return self.green_line_crossed

    def set_green_line_crossed(self, value):
        self.green_line_crossed = value

    def get_digit_prediction(self):
        self.digit_prediction_to_str()
        return self.digit_prediction

    def update_digit_prediction(self, value):
        self.digit_prediction[value] += 1

    def get_most_predicted_digit(self):
        max_times = self.digit_prediction[0]
        digit = 0
        for value in range(1, 9):
            if max_times < self.digit_prediction[value]:
                max_times = self.digit_prediction[value]
                digit = value

        return digit

    def digit_prediction_to_str(self):
        for digit in range(0, 9):
            print("[" + str(digit) + "] = " + str(self.digit_prediction[digit]))


# ovo neka stoji tu implementirano - mozda se iskoristi, zlu ne trebalo
class DigitCoords:
    def __init__(self, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y

    def get_center_x(self):
        return self.center_x
    
    def set_center_x(self, new_center_x):
        self.self_center_x = new_center_x

    def get_center_y(self):
        return self.center_y

    def set_center_y(self, new_center_y):
        self.self_center_y = new_center_y

        

