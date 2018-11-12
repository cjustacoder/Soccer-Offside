import cv2
import numpy as np
import itertools
import random
from itertools import starmap
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import RansacVanishingPoint
import numpy as np
import numpy as np
import random
import time

cap = cv2.VideoCapture("offside1_Trim-2.mp4")




while(1):

    # Take each frame
    _, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)



    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_white = np.array([0,0,150])
    upper_white = np.array([180,100,255])

    lower_green = np.array([51, 0, 0])    ##green field
    upper_green = np.array([60, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask2 = cv2.inRange(hsv, lower_green, upper_green)


    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res2 = cv2.bitwise_and(frame,frame, mask= mask2)



    edges = cv2.Canny(mask, 50, 120)
    edges2 = cv2.Canny(mask2, 50, 120)


    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, lines=100, minLineLength=200, maxLineGap=120)
    # line_img,lines = cv2.HoughLinesP(edges2, 1, np.pi / 180, 80, lines=80, minLineLength=220, maxLineGap=18)

    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         if((x2-x1)<(y2-y1) ):
    #                 cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # print(lines)
    # if lines2 is not None:
    #     for line2 in lines2:
    #         gx1, gy1, gx2, gy2 = line2[0]
    #         if((gx2-gx1)<(gy2-gy1) and (gx2-gx1)>0):
    #                 cv2.line(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)


##------------------------End For Detect Line-------------------##


    def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1, y1, x2, y2 in line:
                if ((x2 - x1) < (y2 - y1)):
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    #
    #
    def hough_lines(img, rho, theta, threshold,
                    min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, lines)
        return line_img, lines


    def roi_mask(img, vertices):
        mask = np.zeros_like(img)
        mask_color = 255
        cv2.fillPoly(mask, vertices, mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img


    roi_vtx = np.array([[(0, frame.shape[0]), (500, 200), (650, 200), (800, frame.shape[0])]])
    roi_edges = roi_mask(edges2, roi_vtx)
    line_img, lines = hough_lines(edges2, 1, np.pi / 180, 80, 220, 18)

    # while(1):
    try:
        Model = RansacVanishingPoint.pointModel()
        (params, inliers, residual, iteration) = RansacVanishingPoint.ransac(lines, Model, 2, eps=1)
        plt.figure(figsize=(15, 10))
        plt.imshow(line_img, cmap='gray')
        print(params[0],params[1])
        break
        # plt.plot(params[0], params[1], '+', markersize=12)
        # plt.show()
    except ValueError:
        pass




    cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask2)
    # cv2.imshow("Edges", edges)
    # cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()