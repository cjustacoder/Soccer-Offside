# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from matplotlib import pyplot as plt

DEBUG = False

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file", default=r'offsidewhite1.mp4')
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())
# print(args)
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)
greenLower = (0, 0, 205)
greenUpper = (255, 50, 255)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)
# fourcc = int(vs.get(cv2.CAP_PROP_FOURCC))
# print("fourcc: ", fourcc)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
print("fourcc: ", fourcc)
fps = vs.get(cv2.CAP_PROP_FPS)

test = vs.get(cv2.CAP_PROP_FOURCC)
size = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('output_test_ball.avi', fourcc, fps, size)
# keep looping
while True:
    # grab the current frame
    frame = vs.read()
    if DEBUG and False:
        print("original frame: ", frame)
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    if DEBUG and False:
        print("changed frame: ", frame)
        time.sleep(10)
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
    if DEBUG and False:
        plt.imshow(frame)
        plt.show()

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=1440)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    if DEBUG:
        plt.subplot(121), plt.imshow(frame), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blurred), plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()
        time.sleep(1)
    if DEBUG:
        # blurred = frame
        pass
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    cv2.imshow("frame", frame)

    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("mask", mask)
    if DEBUG and False:
        print(mask)
        cv2.imshow('mask', mask)
        time.sleep(1)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if DEBUG:
        for i in range(len(cnts)):
            print(cv2.contourArea(cnts[i]))
    # only proceed if at least one contour was found
    cnt_thresh_up = 100
    cnt_thresh_down = 10
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        # c = max(cnts, key=cv2.contourArea)
        for i in range(len(cnts)):
            # print("size of cnt[",i,"] = ",cv2.contourArea(cnts[i]))
            # if cv2.contourArea(cnts[i]) > cnt_thresh_up \
            #         and cv2.contourArea(cnts[i]) < cnt_thresh_down:
            #     print("need to be breaked")
            #     break
            #
            # else:
            #     print("what going in is No.",i)
            #     c = cnts[i]
            c = cnts[i]
            # print("size of c is", cv2.contourArea(c))
            if cv2.contourArea(c) > cnt_thresh_up:
                # print('big')
                continue
            # print("now c is", cv2.contourArea(c))
            if cv2.contourArea(c) < cnt_thresh_down:
                continue
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            # print(M)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center = (0, 0)

            # only proceed if the radius meets a minimum size
            # if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
            # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    cv2.imshow('output', frame)
    out.write(frame)
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break
vs.release()
out.release()
cv2.destroyAllWindows()

# update the points queue
    #         pts.appendleft(center)
