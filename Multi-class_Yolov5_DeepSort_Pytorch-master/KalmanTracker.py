import cv2
import numpy as np
from deep_sort.deep_sort.sort.kalman_filter import *

import math
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'filterpy-master')))
from filterpy.kalman.UKF import *
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
#from filterpy.kalman.kalman_filter import *
#from filterpy-master.filterpy.kalman.CubatureKalmanFilter import *


def hx(x):
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    return np.dot(H, x)

def fx(x, dt):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, dt, 0],
                  [0, 0, 0, dt]])
    return np.dot(F, x)

def InitUKFTracker(measurement_list):
    points = MerweScaledSigmaPoints(n=4, alpha=0.001, beta=2, kappa=0)

    filter_list = []
    x_list = []

    number_of_objects = len(measurement_list)

    for i in range(number_of_objects):
        filter_list.append(UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=1/24, hx=hx, fx=fx, points=points))

    for (i, filter), measurement in zip(enumerate(filter_list), measurement_list):
        center = calculateCenterPoint(measurement)

        filter.x = [center[0], center[1], 0, 0]

        filter.predict()

        x_list.append(filter.x)

    return filter_list, x_list

def predictUKFTracker(filter_list, x_list):
    for i, filter in enumerate(filter_list):
        filter.predict()

        x_list[i] = filter.x
    return filter_list, x_list

def updateUKFTracker(filter_list, x_list, measurement_list):
    for (i, filter), measurement in zip(enumerate(filter_list), measurement_list):
        if measurement != []:

            center = calculateCenterPoint(measurement)

            filter.update(center)

            x_list[i] = filter.x

    return filter_list, x_list

def preprocessMeasurements(measurement_list):
    output_measurement = []  # xyah
    classlist_bboxes = []
    for measurement in measurement_list:
        cls = measurement.pop(-1)
        classlist_bboxes.append(cls)
        center = measurement.pop(-1)
        if cls == 0:
            output_measurement.append([center[0] - 25, center[1] - 25, 1, 50])
        elif cls == 1:
            output_measurement.append([center[0] - 15, center[1] - 15, 1, 30])

    return output_measurement, classlist_bboxes

def calculateCenterPoint(xyah):  # a = w / h
    if xyah != []:
        x1, y1, a, h = xyah[0], xyah[1], xyah[2], xyah[3]

        y2 = y1 + h
        x2 = x1 + a * h

        x = round((x1 + x2) / 2)
        y = round((y1 + y2) / 2)
        return (int(x), int(y))
    else:
        return []

def drawFilterOutput(xyah, frame):
    for i, bbox in enumerate(xyah):
        p = calculateCenterPoint(bbox)

        color = (106,196,255)

        x1, y1 = bbox[0], bbox[1]
        a = 1
        h = 25

        x1 = int(round(x1 - 25))
        y1 = int(round(y1 - 25))
        y2 = int(round(y1 + h))
        x2 = int(round(x1 + a * h))
        cv2.circle(frame, (x1,y1), 1, (0,0,255), 1)
        cv2.circle(frame, (x2, y2), 1, (0, 255, 255), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        #cv2.circle(frame, p, 2, color, 2)

        # ID label
        """label = str(i)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)"""
    return frame

def InitKalmanTracker(measurement_list):
    filter_list = []
    mean_list = []
    covariance_list = []

    number_of_objects = len(measurement_list)

    for i in range(number_of_objects):
        filter_list.append(KalmanFilter())

    for (i, filter), measurement in zip(enumerate(filter_list), measurement_list):
        mean, cov = filter.initiate(measurement)

        mean_list.append(mean)
        covariance_list.append(cov)

    return filter_list, mean_list, covariance_list

def predictKalmanTracker(filter_list, mean_list, cov_list):
    for (i, filter), mean, cov in zip(enumerate(filter_list), mean_list, cov_list):
        meanP, covP = filter.predict(mean, cov)

        mean_list[i] = meanP
        cov_list[i] = covP
    return filter_list, mean_list, cov_list

def updateKalmanTracker(filter_list, mean_list, cov_list, measurement_list):
    for (i, filter), mean, cov, measurement in zip(enumerate(filter_list), mean_list, cov_list, measurement_list):
        if measurement != []:
            meanU, covU = filter.update(mean, cov, measurement)

            mean_list[i] = meanU
            cov_list[i] = covU
        else:
            cov_list[i] = cov
            mean_list[i] = mean
    return filter_list, mean_list, cov_list

def projectKalmanTracker(filter_list, mean_list, cov_list):
    for (i, filter), mean, cov in zip(enumerate(filter_list), mean_list, cov_list):
        meanI, covI = filter.project(mean, cov)
        mean_list[i] = meanI
        cov_list[i] = covI
    return mean_list, cov_list

def association(filter_list, mean_list, measurement_list, class_list):
    # Measurement preprocessing:
    N = len(measurement_list)
    measurement_matrix = np.reshape(measurement_list, (N,4))

    associated_measurement_list = []
    associated_classes = []

    for (i, filter), mean in zip(enumerate(filter_list), mean_list):
        distance = euclidean(mean, measurement_list)
        association_index = np.argmin(distance)

        if distance[association_index] < 100:  # 65
            associated_measurement_list.append(measurement_list[association_index])
            associated_classes.append(class_list[association_index])
        else:
            associated_measurement_list.append([])
    """
    for (i, filter), mean, cov in zip(enumerate(filter_list), mean_list, cov_list):
        squared_maha = filter.gating_distance(mean, cov, measurement_matrix)
        association_index = np.argmin(squared_maha)
        if squared_maha[association_index] < 78:  # 8:
            associated_measurement_list.append(measurement_list[association_index])
        else:
            associated_measurement_list.append([])"""
    return associated_measurement_list, associated_classes


def euclidean(mean, measurement_list):
    distance = []

    x1 = mean[0]
    y1 = mean[1]

    for measurement in measurement_list:
        x2 = measurement[0]
        y2 = measurement[1]
        distance.append(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    return distance

