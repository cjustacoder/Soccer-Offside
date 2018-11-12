import cv2
import numpy as np
import time
import RansacVanishingPoint


# input the test video
cap = cv2.VideoCapture(r'offside1_Trim-2.mp4')

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
pts = []


kernel = np.array([[0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]], dtype=np.uint8)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((x2 - x1) < (y2 - y1)):
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            else:
                continue

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

ret, frame = cap.read()
teamB_new = np.array([])
teamA_new = np.array([])
# x2, y2 = (627, -618)
n = 0
while cap.isOpened():
    n = n+1
    teamA = []
    teamB = []
    # Take each frame
    ret, frame = cap.read()


    # if we get a frame(video is not over)
    if ret:
        # Start time
        start = time.time()
        # End time

        # Time elapsed
        # Convert BGR to HSV
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
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

        lower_green = np.array([51, 0, 0])  ##green field
        upper_green = np.array([60, 255, 255])


        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        # mask_blue = cv2.GaussianBlur(mask_blue, (11, 11), 5)
        # mask_blue = cv2.cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        # mask_blue = cv2.cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        # mask_blue = (mask_blue > 254).sum() / (len(mask_blue) * len(mask_blue[0]))
        # mask_blue = cv2.erode(mask_blue, np.ones((5, 5)))
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        # mask_red = cv2.cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        # mask_red = cv2.cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        # mask_red = (mask_red > 254).sum() / (len(mask_red) * len(mask_red[0]))
        # mask_red = cv2.GaussianBlur(mask_red, (11, 11), 5)
        # mask_red = cv2.dilate(mask_red, kernel)
        # mask_red = cv2.erode(mask_red, np.ones((5, 5)))
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # Bitwise-AND mask and original image
        res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
        res_red = cv2.bitwise_and(frame, frame, mask=mask_red)
        res_white = cv2.bitwise_and(frame, frame, mask=mask_white)
        res_green = cv2.bitwise_and(frame, frame, mask=mask_green)
        # uncommon code below to show video result in window
        # cv2.imshow('frame', frame)
        # cv2.imshow('mask', mask_blue)
        # cv2.imshow('res', res_white)

        # draw bounding box for the players
        cnts_blue = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        edges_green = cv2.Canny(mask_green, 50, 120)
        # retrieves only the extreme outer contours
        # orig_op = cv2.imread('soccer_half_field.jpeg')
        # op = orig_op.copy()
        cnt_thresh = 10
        if len(cnts_blue) > 0 or len(cnts_red) > 0:
            c = sorted(cnts_blue, key=cv2.contourArea, reverse=True)
            d = sorted(cnts_red, key=cv2.contourArea, reverse=True)
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
                teamA.append(foot)
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
                teamB.append(foot)
                cv2.circle(frame, foot, 5, (0, 255, 0), -1)

        roi_vtx = np.array([[(0, frame.shape[0]), (500, 200), (650, 200), (800, frame.shape[0])]])
        # roi_edges = roi_mask(edges2, roi_vtx)
        line_img, lines = hough_lines(edges_green, 1, np.pi / 180, 80, 220, 18)
        try:
            Model = RansacVanishingPoint.pointModel()
            (params, inliers, residual, iteration) = RansacVanishingPoint.ransac(lines, Model, 2, eps=1)
            print(params)
            """find last defender: """
            # if len(teamA) > 0 and n > 105:
            # if len(teamA) > 0:
            #     for i in range(len(teamA)):
            #         x1 = teamA[i][0]
            #         y1 = teamA[i][1]
            #         slope = (params[1]-y1)/(params[0]-x1)
            #         x0 = -y1 / slope + x1
            #         last = 0
            #         if x0 > last:
            #             last = x0
            #             last_def = (x1, y1)
            #         else:
            #             continue
            #     # p = np.array([[last_def], [x2, y2]])
            #     frame = cv2.line(frame, last_def, (int(params[0]), int(params[1])), (255, 0, 0), 2)
        except ValueError:
                    pass
        # end = time.time()
        # # Time elapsed
        # seconds = end - start
        # fps = 1 / seconds
        out.write(frame)
    else:
        break

    if cv2.waitKey(80) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()