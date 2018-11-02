import cv2
import numpy as np

# input the test video
cap = cv2.VideoCapture(r'test.mp4')

# chose a video output encoding method
# fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# set fps of output video(same as input video)
fps = cap.get(cv2.CAP_PROP_FPS)
test = cap.get(cv2.CAP_PROP_FOURCC)

# set size of output video(same as input video)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# generate the output video
out = cv2.VideoWriter('output.avi', fourcc, fps, size)


while cap.isOpened():

    # Take each frame
    ret, frame = cap.read()

    # if we get a frame(video is not over)
    if ret:

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # define range of red color in HSV
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])

        # define range of white color in HSV
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 55, 255])

        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Bitwise-AND mask and original image
        res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
        res_red = cv2.bitwise_and(frame, frame, mask=mask_red)
        res_white = cv2.bitwise_and(frame, frame, mask=mask_white)

        # uncommon code below to show video result in window
        # cv2.imshow('frame', frame)
        # cv2.imshow('mask', mask_blue)
        # cv2.imshow('res', res_white)

        # draw bounding box for the players
        cnts_blue = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # retrieves only the extreme outer contours
        cnt_thresh = 10
        if len(cnts_blue) > 0 or len(cnts_red) > 0:
            c = sorted(cnts_blue, key=cv2.contourArea, reverse=True)
            d = sorted(cnts_red, key=cv2.contourArea, reverse=True)
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

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                M = cv2.moments(c[i])
                # find the center of gravity of the players
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # find the foot of the players
                foot = (center[0], int(center[1] + h*1.1))
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)

            for i in range(len(d)):
                if cv2.contourArea(d[i]) < cnt_thresh:
                    break

                x, y, w, h = cv2.boundingRect(d[i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                M = cv2.moments(d[i])
                # find the center of gravity of the players
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # find the foot of the players
                foot = (center[0], int(center[1] + h * 1.1))
                cv2.circle(frame, foot, 5, (0, 255, 0), -1)

        out.write(frame)
    else:
        break

    if cv2.waitKey(80) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
