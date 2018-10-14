import cv2
import numpy as np


cap = cv2.VideoCapture(r'Copy_of_offsside7.mp4')
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('output.avi', fourcc, fps, size)



while(cap.isOpened()):

    # Take each frame
    ret, frame = cap.read()
    if ret:

    # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)

        # draw bounding box for the players
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt_thresh = 180
        if len(cnts) > 0:
            c = sorted(cnts, key=cv2.contourArea, reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i]) < cnt_thresh:
                    break

                x, y, w, h = cv2.boundingRect(c[i])
                h += 10
                y -= 5
                if h < 0.8 * w:
                    continue
                elif h / float(w) > 3:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                M = cv2.moments(c[i])
                # find the center of gravity of the players
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # find the foot of the players
                foot = (center[0], int(center[1] + h*1.1))
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)

        out.write(frame)
    else:
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

