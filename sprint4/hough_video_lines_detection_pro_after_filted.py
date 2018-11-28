import cv2
import numpy as np

cap = cv2.VideoCapture(r'offsidewhite1.mp4')

while(1):

    # Take each frame
    ret, frame = cap.read()
    if not ret:
        break
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    frame = cv2.filter2D(frame, -1, kernel)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_white = np.array([0, 0, 205])
    upper_white = np.array([255, 50, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)

    edges = cv2.Canny(mask, 50, 120)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, lines=100, minLineLength=500, maxLineGap=200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 180, lines=100, minLineLength=690, maxLineGap=80)
    if lines is not None:
        x1_min = 10000
        x2_min = 10000
        y1_min = 10000
        y2_min = 10000
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if ((x2 - x1)+300 < (y2 - y1)):
                for i in range(len(lines)):
                    if x1<x1_min:
                        x1_min = x1
                        x2_min = x2
                        y1_min = y1
                        y2_min = y2

        cv2.line(frame, (x1_min, y1_min), (x2_min, y2_min), (0, 255, 0), 2)
        print(x1_min,y1_min,x2_min,y2_min)

        # print(lines)
    cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
