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

def ccw(A,B,C):
        return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def check_for_crossing_lines(blue_line, green_line):
    """Proverava da li je presecena linija."""
    blue_line_crossed = False
    green_line_crossed = False
    C = DigitCoords(468, 155)
    D = DigitCoords(542, 245)
    print("first time coords: [ " + str(C.get_x()) + " : " + str(C.get_y()) + " ]")
    print("last time coords: [ " + str(D.get_x()) + " : " + str(D.get_y()) + " ]")
    
    A = DigitCoords(blue_line[0], blue_line[1])
    B = DigitCoords(blue_line[2], blue_line[3])
    print("blue 1. spot coords: [ " + str(A.get_x()) + " : " + str(A.get_y()) + " ]")
    print("blue 2. spot coords: [ " + str(B.get_x()) + " : " + str(B.get_y()) + " ]")           
    if intersect(A,B,C,D):
        blue_line_crossed = True
        
    A = DigitCoords(green_line[0], green_line[1])
    B = DigitCoords(green_line[2], green_line[3])
    print("green 1. spot coords: [ " + str(A.get_x()) + " : " + str(A.get_y()) + " ]")
    print("green 2. spot coords: [ " + str(B.get_x()) + " : " + str(B.get_y()) + " ]")            
    if intersect(A,B,C,D):
        green_line_crossed = True

    return blue_line_crossed, green_line_crossed

blue = (290, 204, 492, 52)
green = (153, 385, 405, 231)

blue_line_crossed, green_line_crossed = check_for_crossing_lines(blue, green)
print("blue_line_crossed: " + str(blue_line_crossed))
print("green_line_crossed: " + str(green_line_crossed))