import cv2
import numpy as np

video = cv2.VideoCapture("C:/Users/14455/Desktop/offside1.mp4")

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("C:/Users/14455/Desktop/offside1.mp4")
        continue

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, lines=100, minLineLength=500, maxLineGap=200)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # print(lines)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()