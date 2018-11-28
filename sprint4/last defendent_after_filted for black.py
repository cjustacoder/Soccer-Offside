
import cv2
import numpy as np
import imutils
from imutils.video import FPS
import time


cap = cv2.VideoCapture(r'offsidenew9.mp4')
# fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
test = cap.get(cv2.CAP_PROP_FOURCC)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('output.avi', fourcc, fps, size)

# create tracker
tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerTLD_create()
# create counter
counter = 0
initBB = None

while(cap.isOpened()):

    # Take each frame
    ret, frame = cap.read()
    if ret:
        # -------------------bounderay line----------------------------------
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # sharpen = cv2.filter2D(frame, -1, kernel)
        # # blurred = cv2.GaussianBlur(sharpen, (5, 5), 0)
        # # Convert BGR to HSV
        # hsv = cv2.cvtColor(sharpen, cv2.COLOR_BGR2HSV)
        #
        # # define range of blue color in HSV
        # lower_white = np.array([0, 0, 150])
        # upper_white = np.array([255, 50, 255])
        #
        # # Threshold the HSV image to get only blue colors
        # mask = cv2.inRange(hsv, lower_white, upper_white)
        #
        # # Bitwise-AND mask and original image
        # res = cv2.bitwise_and(frame, frame, mask=mask)
        #
        # edges = cv2.Canny(mask, 50, 120)
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, lines=100, minLineLength=1000, maxLineGap=200)
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         if (x2 - x1) < (y2 - y1):
        #             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(frame, -1, kernel)
        # Convert BGR to HSV
        hsv = cv2.cvtColor(sharpen, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_white_line = np.array([0, 0, 205])
        upper_white_line = np.array([255, 50, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_white_line, upper_white_line)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        edges = cv2.Canny(mask, 50, 120)
        ##offsidewhite
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 180, lines=100, minLineLength=690, maxLineGap=120)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 165, lines=100, minLineLength=675, maxLineGap=120)

        if lines is not None:
            x1_min = 10000
            x2_min = 10000
            y1_min = 10000
            y2_min = 10000
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if ((x2 - x1) + 300 < (y2 - y1)):
                    for i in range(len(lines)):
                        if x1 < x1_min:
                            x1_min = x1
                            x2_min = x2
                            y1_min = y1
                            y2_min = y2


            # cv2.line(frame, (x1_min, y1_min), (x2_min, y2_min), (0, 255, 0), 2)
            # print(x1_min, y1_min, x2_min, y2_min)
    # -----------------player detection --------------------
    # Convert BGR to HSV
        blurred = cv2.GaussianBlur(frame, (13, 13), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # # define range of blue color in HSV
    #     lower_blue = np.array([100, 0, 0])
    #     upper_blue = np.array([140, 255, 255])
    # # define range of red color in HSV
    #     lower_red = np.array([0, 0, 0])
    #     upper_red = np.array([5, 255, 255])

    # define range of white color in HSV
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 55, 255])
    # define range of yellow color in HSV
        lower_yellow = np.array([33, 50, 70])
        upper_yellow = np.array([40, 150,170])

    ##-----------test for offsidenew9 video-------------------##
    # # define range of red color for second video in HSV
        lower_red = np.array([0, 0, 0])
        upper_red = np.array([10, 255, 255])
    # define range of blue color in HSV
        lower_blue = np.array([100, 0, 0])
        upper_blue = np.array([180, 150, 155])

    ##————————————test for  offsidenew2 video--------------##
    # define range of blue color in HSV
    #     lower_blue = np.array([100, 0, 0])
    #     upper_blue = np.array([180, 255, 255])
    # # define range of red color in HSV
    #     ##black color attacker
    #     lower_red = np.array([0, 0, 65])
    #     upper_red = np.array([180, 150 , 82])
    #     # lower_red = np.array([0, 50, 65])
    #     # upper_red = np.array([180, 180, 85])

        # opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        # hsv2 = cv2.cvtColor(opening, cv2.COLOR_BGR2HSV)


        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)


        # Bitwise-AND mask and original image
        res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
        res_red = cv2.bitwise_and(frame, frame, mask=mask_red)
        res_white = cv2.bitwise_and(frame, frame, mask=mask_white)
        res_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)

        # show res in window
        # cv2.imshow('frame', frame)
        # cv2.imshow('mask', mask_blue)
        cv2.imshow('mask2', mask_red)
        # cv2.imshow('res', res_white)

        # draw bounding box for the players
        cnts_blue = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts_white = cv2.findContours(mask_white.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts_yellow = cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # retrieves only the extreme outer contours
        cnt_thresh = 100
        location = []
        if len(cnts_blue) > 0:
            c = sorted(cnts_blue, key=cv2.contourArea, reverse=True)
            d = sorted(cnts_white, key=cv2.contourArea, reverse=True)
            e = sorted(cnts_red, key=cv2.contourArea, reverse=True)
            f = sorted(cnts_yellow, key=cv2.contourArea, reverse=True)

            for i in range(len(c)):

                if cv2.contourArea(c[i]) < cnt_thresh:
                    break

                x, y, w, h = cv2.boundingRect(c[i])

                # h += 10
                # y -= 5
                # if h < 0.8 * w:
                #     continue
                # elif h / float(w) > 3:
                #     continue
                # if x1_min<x:
                #     pass
                # else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                M = cv2.moments(c[i])
                # find the center of gravity of the players
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    center = (0, 0)
                # find the foot of the players
                location.append(center[0])
                foot = (center[0], int(center[1] + h*1.1))
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)
            # find the last one
            # print("location of defendent: ",location)
            # print("maximum is :", max(location))
            # print("index: ", location.index(max(location)))
            cv2.line(frame, (max(location), 0), (max(location), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (255, 0, 0), 2)
            ##yellow color attacker
            # for i in range(len(f)):
            #     if cv2.contourArea(f[i]) < cnt_thresh:
            #         break
            #
            #     x, y, w, h = cv2.boundingRect(f[i])
            #     if x1_min<x:
            #         pass
            #     else:
            #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #         M = cv2.moments(f[i])
            #         # find the center of gravity of the players
            #         # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #         if M["m00"] != 0:
            #             center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #         else:
            #             center = (0, 0)
            #         # find the foot of the players
            #         foot = (center[0], int(center[1] + h * 1.1))
            #         cv2.circle(frame, foot, 5, (0, 255, 0), -1)


            ## red color attacker
            for i in range(len(e)):
                print(cv2.contourArea(e[i]))
                if cv2.contourArea(e[i]) < cnt_thresh:
                    break

                x, y, w, h = cv2.boundingRect(e[i])
                # if x1_min<x:
                #     pass
                # else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                M = cv2.moments(e[i])
                # find the center of gravity of the players
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    center = (0, 0)
                # find the foot of the players
                foot = (center[0], int(center[1] + h * 1.1))
                cv2.circle(frame, foot, 5, (0, 255, 0), -1)
            print("==========================================")
        # ----------------player detection end---------------------------------------------
        # ----------------ball tracking ---------------------------------------------------
        # ------here I want to set auto detect ball to help tracking--------------
        flag_track = False
        if counter % 10 == 0:
            # print("bingo")
            # library of color
            whiteLower = (0, 0, 200)
            whiteUpper = (255, 50, 255)
            #new video ball color
            # whiteLower = (0, 0, 200)
            # whiteUpper = (155, 55, 255)
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
            # print(cnts)
            # set threshold
            cnt_thresh_up = 50
            cnt_thresh_down = 10
            if len(cnts) > 0:
                # print("have cnts")
                for i in range(len(cnts)):
                    c = cnts[i]
                    # print(cv2.contourArea(c))
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
                    if center:
                        break
        # print(center)
            if center:
                initBB = (center[0], center[1], 15, 15)
                fps = FPS().start()
                tracker.init(frame, initBB)
                flag_track = True
                # print("initBB",initBB)
    # -----------------------------------------------------------------
    # check to see if we are currently tracking an object
    # print(initBB)
        if initBB is not None:
            if flag_track:
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, initBB)
                flag_track = False
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)


            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                # print("box",box)
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (255, 255, 0), 2)

            # update the FPS counter
            fps.update()
            fps.stop()

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                # ("Tracker", "MIL"),
                ("Tracker", "TLD"),
                ("Success", "Yes" if success else "No"),
                # ("FPS", "{:.2f}".format(fps.fps())),
            ]

            # loop over the info tuples and draw them on our frame
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # show the output frame
        cv2.imshow("Frame", frame)
        # time.sleep(0.1)
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


# every thing is done write to screen
        out.write(frame)
    else:
        break

    if cv2.waitKey(80) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()