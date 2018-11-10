import cv2
import numpy as np
import itertools
import random
from itertools import starmap

cap = cv2.VideoCapture("offside1.mp4")

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None  # Lines don't cross

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return x, y

def find_intersections(lines,lines2):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in enumerate(lines2):
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:  # If lines cross, then add
                    intersections.append(intersection)

    return intersections


# Given intersections, find the grid where most intersections occur and treat as vanishing point
def find_vanishing_point(img, grid_size, intersections):
    # Image dimensions
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Grid dimensions
    grid_rows = (image_height // grid_size) + 1
    grid_columns = (image_width // grid_size) + 1

    # Current cell with most intersection points
    max_intersections = 0
    best_cell = (0.0, 0.0)

    for i, j in itertools.product(range(grid_rows), range(grid_columns)):
        cell_left = i * grid_size
        cell_right = (i + 1) * grid_size
        cell_bottom = j * grid_size
        cell_top = (j + 1) * grid_size
        cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 10)

        current_intersections = 0  # Number of intersections in the current cell
        for x, y in intersections:
            if cell_left < x < cell_right and cell_bottom < y < cell_top:
                current_intersections += 1

        # Current cell has more intersections that previous cell (better)
        if current_intersections > max_intersections:
            max_intersections = current_intersections
            best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)
            print("Best Cell:", best_cell)

    if best_cell[0] != None and best_cell[1] != None:
        rx1 = int(best_cell[0] - grid_size / 2)
        ry1 = int(best_cell[1] - grid_size / 2)
        rx2 = int(best_cell[0] + grid_size / 2)
        ry2 = int(best_cell[1] + grid_size / 2)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 10)
        cv2.imwrite('/pictures/output/center.jpg', img)

    return best_cell


while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_white = np.array([0,0,150])
    upper_white = np.array([180,100,255])

    lower_green = np.array([51, 120, 0])    ##green field
    upper_green = np.array([60, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask2 = cv2.inRange(hsv, lower_green, upper_green)


    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res2 = cv2.bitwise_and(frame,frame, mask= mask2)


    edges = cv2.Canny(mask, 50, 120)
    edges2 = cv2.Canny(mask2, 50, 120)


    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, lines=100, minLineLength=200, maxLineGap=120)
    lines2 = cv2.HoughLinesP(edges2, 1, np.pi / 180, 100, lines=100, minLineLength=10, maxLineGap=120)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if((x2-x1)<(y2-y1) ):
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # print(lines)
    if lines2 is not None:
        for line2 in lines2:
            gx1, gy1, gx2, gy2 = line2[0]
            if((gx2-gx1)<(gy2-gy1) and (gx2-gx1)>0):
                    cv2.line(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)


    # line_intersection(line, line2)
    # find_intersections(lines)
    # find_vanishing_point(img, grid_size, intersections)


    cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    # cv2.imshow("Edges", edges)
    # cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()