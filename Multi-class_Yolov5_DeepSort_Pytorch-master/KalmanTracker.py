import cv2
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'GitHub/filterpy-master')))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from filterpy.kalman import *
from filterpy.kalman import MerweScaledSigmaPoints

""" 
Part of Master Thesis 'Indoor Tracking using a Central Camera System' at Chalmers University of Technology, conducted
at Sigma Technology Insights 2021.

Authors:
Jonas Lindberg
Sara Roth

"""

def hx(x):
    """ Converts state vector x into a measurement vector
        Measurement matrix H
        What is measured and how it relates to the state vector
    """
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    return np.dot(H, x)

def fx(x, dt):
    """ The Dynamic matrix helps us in defining the equations for predicting the Vehicle Motion Model
        returns the state x transformed by the state transition function.
        dt is the time step in seconds
    """
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, dt, 0],
                  [0, 0, 0, dt]])
    return np.dot(F, x)

def InitUKFTracker(measurement_list, cls_list):
    """ Initialize UKF tracker
    Input:  measurement_list: List with each measurement
            cls_list: List with classes to each measurement

    Output: filter_list: List with UKF filterobjects  for each object
            states_list: List with all state variables
    """
    # Generate sigma points
    points = MerweScaledSigmaPoints(n=4, alpha=0.001, beta=2, kappa=0)

    filter_list = []
    states_list = []

    number_of_objects = len(measurement_list)
    for i in range(number_of_objects):
        # Add an UKF-object for each object in a list
        filter_list.append(UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=1/24, hx=hx, fx=fx, points=points,
                                                 cls=cls_list[i], id=i))

    for (i, filter), measurement in zip(enumerate(filter_list), measurement_list):
        center = calculateCenterPoint(measurement)
        # Number of state variables for the filter
        filter.x = [center[0], center[1], 0, 0]
        # Performs the predict step of the UKF
        filter.predict()
        # Each state variables is appended to a states_list
        states_list.append(filter.x)

    return filter_list, states_list

def predictUKFTracker(filter_list, states_list):
    """ Prediction step  UKF tracker
    Input:  measurement_list: List with each measurement
            cls_list: List with classes to each measurement

    Output: filter_list: List with UKF filterobjects  for each object
            states_list: List with all state variables
    """

    cls_list = []

    for i, filter in enumerate(filter_list):
        filter.predict()

        states_list[i] = filter.x
        cls_list.append(filter.cls)

    return filter_list, states_list, cls_list

def updateUKFTracker(filter_list, states_list, measurement_list):
    """ Update step  UKF tracker. Calculate a center point based on the measurement
    Input:  filter_list: List with UKF objects for each object
            measurement_list: List with each measurement
            states_list: List with all state variables

    Output: filter_list: List with UKF filterobjects for each object
            states_list: List with all state variables
    """

    for (i, filter), measurement in zip(enumerate(filter_list), measurement_list):
        if measurement != []:

            center = calculateCenterPoint(measurement)

            filter.update(center)

            states_list[i] = filter.x

    return filter_list, states_list

def preprocessMeasurements(measurement_list):
    """ Preprocess measurements to get the output in [TLx, TLy, a, h]
    Input:  measurement_list: List with each measurement [[CPx, CPy], class]

    Output: output_measurement: List with measurements for each box in [TLx, TLy, a, h]
            classlist_bboxes: List with classes for each measurement
    """

    output_measurement = []
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
    """ Calculates the centerpoint
    Input:  xyah: [TLx, TLy, a, h]

    Output: Outputs the centerpoint CPx and CPy
    """
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
    """ Draws the filtered output on frame
    Input:  xyah: [TLx, TLy, a, h]
            frame: Frame to draw boxes on

    Output: frame with the drawn bboxes
    """
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

        # ID label
        """label = str(i)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)"""
    return frame

def association(filter_list, mean_list, measurement_list, class_list):
    """ Calculates the distance between mean and measurenetns and associate measurements with tracker
    Input:  filter_list: List with UKF filterobjects  for each object
            mean_list: List with mean in x and y for each object
            measurement_list: List with measurement x and y for each object
            class_list: List with classes for each object

    Output: associated_measurement_list: List with the order of the measurements in the order they are associated to the trackers
    """
    # Measurement preprocessing:
    nbr_measurements = len(measurement_list)
    nbr_trackers = len(filter_list)

    associated_measurement_list = [[] for i in range(nbr_trackers)]
    associated_measurement_matrix = []
    for mean in mean_list:
        # Calculate the euclidean distance between mean and measurement
        distance = euclidean(mean, measurement_list)
        associated_measurement_matrix.append(distance)

    associated_measurement_matrix = np.reshape(associated_measurement_matrix,(nbr_trackers, nbr_measurements))

    for i in range(nbr_measurements):
        col = associated_measurement_matrix[:,i]
        # Choose the index with the smallest distance between measurement and mean
        association_index = np.argmin(col)
        distance_between_state_measurement = col[association_index]
        # Longest distance allowed between associated state and measurement
        if distance_between_state_measurement < 100:
            # Associate the measurement with the smallest distance to the mean in a list
            associated_measurement_list[association_index] = measurement_list[i]

    return associated_measurement_list #, associated_classes


def euclidean(mean, measurement_list):
    """ Calculates the distance between mean and measurenetns
    Input:  mean: mean in  and y
            measurement_list: List with measurement x and y for each object

    Output: distance: Distamce between measurement and mean
    """
    distance = []

    x1 = mean[0]
    y1 = mean[1]

    for measurement in measurement_list:
        x2 = measurement[0]
        y2 = measurement[1]
        distance.append(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    return distance

