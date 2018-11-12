import cv2
import numpy as np
import itertools
import random
from itertools import starmap
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import RansacVanishingPoint


cap = cv2.VideoCapture("offside1_Trim-2.mp4")

import numpy as np
import random

import numpy as np
import random

import random
# import Ransac
import time

num_iterations = 500
num_samples = 1000
noise_ratio = 0.8
num_noise = int(noise_ratio * num_samples)


class Model(object):
    def fit(self, data):
        raise NotImplementedError

    def distance(self, data):
        raise NotImplementedError


class LineModel(Model):
    """
    Multi 2D line model.
    """

    def __init__(self, num):
        self.params = None
        self.dists = None
        self.num = num

    def fit(self, data):
        """
        Fits the model to the data, minimizing the sum of absolute errors.
        """
        X = data[:, 0]
        Y = data[:, 1]
        #        denom = (X[-1] - X[0])
        #        if denom == 0:
        #            raise ZeroDivisionError
        #        k = (Y[-1] - Y[0]) / denom
        #        m = Y[0] - k * X[0]
        A = np.vstack([X, np.ones(len(X))]).T
        k, m = np.linalg.lstsq(A, Y, None)[0]
        self.params = [k, m]
        self.residual = sum(abs(k * X + m - Y))

    def distance(self, samples):
        """
        Calculates the vertical distances from the samples to the model.
        """
        X = samples[:, 0]
        Y = samples[:, 1]
        k = self.params[0]
        m = self.params[1]
        dists = abs(k * X + m - Y)
        #        dists = abs(-k * X + Y - m) / math.sqrt(k**2 + 1)

        return dists


def seqRansac(data, model, min_samples, min_inliers, num_iterations=100, eps=1e-10, random_seed=42):
    """
    Fits a model to observed data.

    Uses the RANSC iterative method of fitting a model to observed data.
    """
    random.seed(random_seed)

    if len(data) <= min_samples:
        raise ValueError("Not enough input data to fit the model.")

    if 0 < min_inliers and min_inliers < 2:
        min_inliers = int(min_inliers * len(data))

    params = []
    inliers = []
    residuals = []
    iterations = []

    for i in range(model.num):

        best_params = None
        best_inliers = None
        best_residual = np.inf
        best_iteration = None

        for i in range(num_iterations):
            _indices = list(range(len(data)))
            random.shuffle(_indices)
            _inliers = np.asarray([data[i] for i in _indices[:min_samples]])
            shuffled_data = np.asarray([data[i] for i in _indices[min_samples:]])

            try:
                model.fit(_inliers)
                dists = model.distance(shuffled_data)
                more_inliers = shuffled_data[np.where(dists <= eps)[0]]
                _inliers = np.concatenate((_inliers, more_inliers))

                if len(_inliers) >= min_inliers:
                    model.fit(_inliers)
                    if model.residual < best_residual:
                        best_params = model.params
                        best_inliers = _inliers
                        best_residual = model.residual
                        best_iteration = i

            except ZeroDivisionError as e:
                print(e)

        if best_params:
            params.append(best_params)
            inliers.append(best_inliers)
            residuals.append(best_residual)
            iterations.append(best_iteration)
            data = np.delete(data, [i for i in best_inliers], axis=0)

    if params is None:
        raise ValueError("Sequential RANSAC failed to find a sufficiently good fit for the data.")
    else:
        return (params, inliers, residuals, iterations)
#
#

#
#







# class Model(object):
#     def fit(self, data):
#         raise NotImplementedError
#
#     def distance(self, data):
#         raise NotImplementedError
#
#
# class Line(object):
#
#     def init(self, line):
#
#         self.x1 = line[0]
#
#         self.y1 = line[1]
#
#         self.x2 = line[2]
#
#         self.y2 = line[3]
#
# def getlineparam(line):
#     """
#     For the line equation a*x+b*y+c=0, if we know two points(x1, y1)(x2,y2) in line, we can get
#         a = y1 - y2
#         b = x2 - x1
#         c = x1*y2 - x2*y1
#     """
#     a = line.y1 - line.y2
#     b = line.x2 - line.x1
#     c = line.x1 * line.y2 - line.x2 * line.y1
#     return a,b,c
#
# def getcrosspoint(line1,line2):
#     """
#     if we have two lines: a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0,
#     when d(= a1 * b2 - a2 * b1) is zero, then the two lines are coincident or parallel.
#     The cross point is :
#         x = (b1 * c2 - b2 * c1) / d
#         y = (a2 * c1 - a1 * c2) / d
#     """
#     a1, b1, c1 = getlineparam(line1)
#     a2, b2, c2 = getlineparam(line2)
#     d = a1 * b2 - a2 * b1
#     if d == 0:
#         return np.inf, np.inf
#     x = (b1 * c2 - b2 * c1) / d
#     y = (a2 * c1 - a1 * c2) / d
#     return x, y
#
#
# class pointModel(Model):
#     def __init__(self):
#         self.params = None
#         self.residual = 0
#
#     def fit(self, lines):
#         lines = lines[:, 0]
#         X = []
#         Y = []
#
#         for i in range(len(lines) - 1):
#             line1 = Line(lines[i])
#             line2 = Line(lines[i + 1])
#             x, y = getcrosspoint(line1, line2)
#             X.append(x)
#             Y.append(y)
#
#         X = np.asarray(x).mean()
#         Y = np.asarray(y).mean()
#         self.params = [X, Y]
#
#         for i in range(len(lines)):
#             line = Line(lines[i])
#             a, b, c = getlineparam(line)
#             self.residual += abs(a * X + b * Y + c) / np.sqrt(a * a + b * b)
#
#     def distance(self, samplelines):
#         dists = []
#         for line in samplelines:
#             line = Line(line[0, :])
#             [x, y] = self.params
#             a, b, c = getlineparam(line)
#             dist = abs((a * x + b * y + c) / np.sqrt(a * a + b * b))
#             dists.append(dist)
#         return np.asarray(dists)
#
#
# def ransac(lines, model, min_samples, min_inliers, iterations=100, eps=1, random_seed=42):
#     """
#     Fits a model to observed data.
#
#     Uses the RANSC iterative method of fitting a model to observed data.
#     """
#     random.seed(random_seed)
#
#     if len(lines) <= min_samples:
#         raise ValueError("Not enough input lines to fit the model.")
#
#     if 0 < min_inliers and min_inliers < 1:
#         min_inliers = int(min_inliers * len(lines))
#
#     best_params = None
#     best_inliers = None
#     best_residual = np.inf
#     best_iteration = None
#
#     for i in range(iterations):
#         indices = list(range(len(lines)))
#         random.shuffle(indices)
#         inliers = np.asarray([lines[i] for i in indices[:min_samples]])
#         shuffled_lines = np.asarray([lines[i] for i in indices[min_samples:]])
#         try:
#             model.fit(inliers)
#             dists = model.distance(shuffled_lines)
#             more_inliers = shuffled_lines[np.where(dists <= eps)[0]]
#             inliers = np.concatenate((inliers, more_inliers))
#
#             if len(inliers) >= min_inliers:
#                 model.fit(inliers)
#                 if model.residual < best_residual:
#                     best_params = model.params
#                     best_inliers = inliers
#                     best_residual = model.residual
#                     best_iteration = i
#
#         except ZeroDivisionError as e:
#             print(e)
#
#     if best_params is None:
#         raise ValueError("RANSAC failed to find a sufficiently good fit for the data.")
#     else:
#         return (best_params, best_inliers, best_residual, best_iteration)

def setup():
    global ax1
    num = 2
    X = np.asarray(range(num_samples))
    Y1 = 1 * X
    Y2 = 2 * X
    Y = np.asarray([Y1[i] if i % 2!=0 else Y2[i] for i in range(num_samples)])
    noise = [random.randint(0, 2 * (num_samples - 1)) for i in range(num_noise)]
    Y[random.sample(range(len(Y)), num_noise)] = noise
    data = np.asarray([X, Y]).T
    model = LineModel(num)
    ax1 = plt.subplot(1,2,1)
    plt.plot(X, Y, 'bx')
    return data, model

def run(data, model):
    random_seed = random.randint(0,100)
    start_time = time.time()
    (params, inliers, residuals, iterations) = seqRansac(data, model, 2, 0.08, num_iterations, 1e-10, random_seed)
    end_time = time.time()
    mean_time = (end_time - start_time) / num_iterations
    return params, residuals, mean_time, iterations


def summary(params, residual, mean_time, iterations):
    print(" Paramters ".center(40, '='))
    print(params)
    print(" Residual ".center(40, '='))
    print(residual)
    print(" Iterations ".center(40, '='))
    print(iterations)
    print(" Time ".center(40, '='))
    print("%.1f msecs mean time spent per call" % (1000 * mean_time))
    X = np.asanyarray([0, num_samples - 1])
    plt.subplot(1, 2, 2, sharey=ax1)
    for param in params:
        plt.plot(X, param[0] * X + param[1], 'y-')

    plt.show()

while(1):

    # Take each frame
    _, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)



    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_white = np.array([0,0,150])
    upper_white = np.array([180,100,255])

    lower_green = np.array([51, 0, 0])    ##green field
    upper_green = np.array([60, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask2 = cv2.inRange(hsv, lower_green, upper_green)


    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res2 = cv2.bitwise_and(frame,frame, mask= mask2)



    edges = cv2.Canny(mask, 50, 120)
    edges2 = cv2.Canny(mask2, 50, 120)


    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, lines=100, minLineLength=200, maxLineGap=120)
    # line_img,lines = cv2.HoughLinesP(edges2, 1, np.pi / 180, 80, lines=80, minLineLength=220, maxLineGap=18)

    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         if((x2-x1)<(y2-y1) ):
    #                 cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # print(lines)
    # if lines2 is not None:
    #     for line2 in lines2:
    #         gx1, gy1, gx2, gy2 = line2[0]
    #         if((gx2-gx1)<(gy2-gy1) and (gx2-gx1)>0):
    #                 cv2.line(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)



    # plt.show()
    # Model = pointModel()
    # Model = LineModel()
    # (params, inliers, residual, iteration) = ransac(lines, Model, min_samples=2, min_inliers=500, iterations=100, eps=1, random_seed=42)
    # data, model = setup()

    # params, residual, mean_time, iterations = run(data, model)


    # cv2.circle(frame, (100, 100), 20, (0, 0, 255), 8)

    # summary(params, residual, mean_time, iterations)
    #
    # plt.figure(figsize=(15, 10))
    # plt.imshow(mask2, cmap='gray')
    # plt.show()
    #
    #
    def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1, y1, x2, y2 in line:
                if ((x2 - x1) < (y2 - y1)):
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
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


    def roi_mask(img, vertices):
        mask = np.zeros_like(img)
        mask_color = 255
        cv2.fillPoly(mask, vertices, mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img


    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_length = 100
    max_line_gap = 20

    roi_vtx = np.array([[(0, frame.shape[0]), (500, 200), (650, 200), (800, frame.shape[0])]])
    roi_edges = roi_mask(edges2, roi_vtx)
    # line_img, lines = hough_lines(edges2, 1, np.pi / 180, 80, 10, 120)
    line_img, lines = hough_lines(edges2, 1, np.pi / 180, 80, 220, 18)

    while(1):
        try:
            Model = RansacVanishingPoint.pointModel()
            (params, inliers, residual, iteration) = RansacVanishingPoint.ransac(lines, Model, 2, eps=1)
            plt.figure(figsize=(15, 10))
            plt.imshow(line_img, cmap='gray')
            print(params[0],params[1])
            break
            # plt.plot(params[0], params[1], '+', markersize=12)
            # plt.show()
        except ValueError:
            pass




    # cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask2)
    # cv2.imshow("Edges", edges)
    # cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()