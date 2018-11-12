# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default=r'offsidewhite1.mp4',
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="mil",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())
# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
# print("major: ", major, "minor: ", minor)
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None
counter = 0

# this part use to 
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=1440)
    (H, W) = frame.shape[:2]
    # -------------------------here I want to set auto detect ball to help tracking--------------
    if counter % 20 == 0:
        # library of color
        whiteLower = (0, 0, 205)
        whiteUpper = (255, 50, 255)
        # blur to help detect
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # change color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # create mask regarding to color
        mask = cv2.inRange(hsv, whiteLower, whiteUpper)
        cv2.imshow("mask", mask)
        # extract out contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None
        # sort contours
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # set threshold
        cnt_thresh_up = 50
        cnt_thresh_down = 10
        if len(cnts) > 0:
            # print("have cnts")
            for i in range(len(cnts)):
                c = cnts[i]
                print(cv2.contourArea(c))
                if cv2.contourArea(c) > cnt_thresh_up:
                    continue
                if cv2.contourArea(c) < cnt_thresh_down:
                    continue
                # print("have something pass")
                M = cv2.moments(c)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    center = (0, 0)
        print(center)
        if center:
            initBB = (center[0], center[1], 15, 15)
            fps = FPS().start()
            tracker.init(frame, initBB)
            print(initBB)
    # -----------------------------------------------------------------
    # check to see if we are currently tracking an object
    # print(initBB)
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            # ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    time.sleep(0.1)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)
        # print(initBB)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
        # fps = vs.get(cv2.CAP_PROP_FPS)
        # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
    counter = counter + 1
# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()

