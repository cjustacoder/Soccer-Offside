# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
from collections import deque
from imutils.video import VideoStream



# core code of image sharpen
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file", default=r'offsidewhite1.mp4')
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

lower_green = np.array([45, 145, 0])    ##New video green field
upper_green = np.array([48, 255, 255])

pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)
while(True):
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=1440)
    (H, W) = frame.shape[:2]
# Reading in and displaying our image
# image = cv2.imread('images/input.jpg')
# cv2.imshow('Original', image)

    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])

# applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(frame, -1, kernel_sharpening)

    cv2.imshow('Image Sharpening', sharpened)

    cv2.waitKey(0)
cv2.destroyAllWindows()
# applying...
