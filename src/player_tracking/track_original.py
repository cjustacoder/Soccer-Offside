import cv2
import numpy as np


cap = cv2.VideoCapture(r'Copy_of_offsside7.mp4')
# fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('output.avi', fourcc,fps , size)
# out = cv2.VideoWriter('output.mp4',fourcc,fps,(800,600))

# writer = VideoWriter('output.mp4', frameSize=size)

while(cap.isOpened()):

    # Take each frame
    ret, frame = cap.read()
    if ret == True:

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
        out.write(res)
    else:
        break
        
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
# out.release()
cv2.destroyAllWindows()

