import cv2

from deep_sort.deep_sort.sort.kalman_filter import KalmanFilter

def calculateCenterPoint(xyah):  # a = w / h
    x1, y1, a, h = xyah[0], xyah[1], xyah[2], xyah[3]

    y2 = y1 + h
    x2 = x1 + a * h

    x = x1 + x2 / 2
    y = y1 + y2 / 2
    return (x, y)

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
        meanU, covU = filter.update(mean, cov, measurement)

        mean_list[i] = meanU
        cov_list[i] = covU
    return filter_list, mean_list, cov_list




