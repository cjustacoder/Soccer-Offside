import cv2
import numpy as np
videoCapture = cv2.VideoCapture('C:/Users/14455/Desktop/offsidenew2.mp4')  # read from video
# judge whether the video is opened
if (videoCapture.isOpened()):
    print
    'Open'
else:
    print
    'Fail to open!'

fps = videoCapture.get(cv2.CAP_PROP_FPS)  # get the fps from the original video

size = (int(600), int(650))  # the size to cut from the original video
# size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))#get the size of the original video
videoWriter = cv2.VideoWriter('C:/Users/14455/Desktop/offsidenew2.avi', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
success, frame = videoCapture.read()  # read from the first frame

while success:
    frame = frame[0:650, 1200:1800]  # cut and get the frame
    videoWriter.write(frame)  # Write the captured picture to "New Video"
    success, frame = videoCapture.read()  # get the next frame repeatedly
videoCapture.release()