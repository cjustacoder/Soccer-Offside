import cv2
import numpy as np

cap = cv2.VideoCapture(r'offsidewhite1.mp4')

while(1):

    # Take each frame
    _, frame = cap.read()

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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, lines=100, minLineLength=500, maxLineGap=200)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # print(lines)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
