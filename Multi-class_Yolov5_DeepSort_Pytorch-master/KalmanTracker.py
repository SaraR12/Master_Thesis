import cv2
import numpy as np

from deep_sort.deep_sort.sort.kalman_filter import *

def calculateCenterPoint(xyah):  # a = w / h
    x1, y1, a, h = xyah[0], xyah[1], xyah[2], xyah[3]

    y2 = y1 + h
    x2 = x1 + a * h

    x = round((x1 + x2) / 2)
    y = round((y1 + y2) / 2)
    return (int(x), int(y))

def drawFilterOutput(xyah, frame):
    for bbox in xyah:
        p = calculateCenterPoint(bbox)
        cv2.circle(frame, p, 2, (0,0,255), 2)
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

def association(filter_list, mean_list, cov_list, measurement_list):
    # Measurement preprocessing:
    N = len(measurement_list)
    measurement_matrix = np.reshape(measurement_list, (N,4))

    associated_measurement_list = []

    for (i, filter), mean, cov in zip(enumerate(filter_list), mean_list, cov_list):
        squared_maha = filter.gating_distance(mean, cov, measurement_matrix)
        association_index = np.argmin(squared_maha)
        if squared_maha[association_index] < 8:
            associated_measurement_list.append(measurement_list[association_index])
        else:
            associated_measurement_list.append([])
    return associated_measurement_list


