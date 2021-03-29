from groundTruth import *
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Convert pixels to meters
def pixel_to_meter(state):
    return [state[0]*50/1788, (1069 - state[1]) * 30/1069]
def pixel_to_meterEval(state):
    return [state[0]*26/1080, (1080 - state[1]) * 26/1080]

# Get the true position
"""AGV1Meter, AGV2Meter, AGV3Meter, AGV4Meter, AGV5Meter = getGroundTruthEvaluation()
true_position_list = [AGV4Meter, AGV2Meter, AGV1Meter, AGV3Meter, AGV5Meter]"""
AGV1Meter = getGroundTruthOneAGV()
true_position_list = [AGV1Meter]
agv1error, agv2error, agv3error, agv4error, agv5error = [], [], [], [], []
errors = [agv1error]
errorx = []
errory = []
names_list = ['AGV1Meter', 'AGV2Meter', 'AGV3Meter', 'AGV4Meter', 'AGV5Meter']

names_list = ['AGV1Meter']
# Get the filtered position
combined_total_error = []
def filtered_positions(states, frame):
    for state, truth, names, error in zip(states, true_position_list, names_list, errors):

        filter_x = pixel_to_meterEval(state)[0]
        filter_y = pixel_to_meterEval(state)[1]
        true_x = truth[frame-1][0]
        try:
            true_y = truth[frame-1][1]
        except:
            print('bai')
        total_error_x = abs(filter_x - true_x)
        total_error_y = abs(filter_y - true_y)
        total_error = math.sqrt(total_error_x**2 + total_error_y**2)
        error.append([total_error])
        errorx.append(total_error_x)
        errory.append(total_error_y)

        combined_total_error.append(total_error)

        print(names)
        print('Filtered ', pixel_to_meterEval(state))
        print('True', truth[frame-1])
        print('Error x', total_error_x )
        print('Error y', total_error_y)
        print('Total error', total_error)
        print('-----------------------------------------')
    print('TOTAL ERROR ', np.mean(combined_total_error))
    if frame == 299:
        print('Mean error AGV1 = ', np.mean(agv1error))
        """print('Mean error AGV2 = ', np.mean(agv2error))
        print('Mean error AGV3 = ', np.mean(agv3error))
        print('Mean error AGV4 = ', np.mean(agv4error))
        print('Mean error AGV5 = ', np.mean(agv5error))"""
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
        fig.suptitle('Positioning error (meter)')
        ax1.plot(range(len(agv1error)), agv1error)
        ax2.plot(range(len(errorx)), errorx)
        ax3.plot(range(len(errory)), errory)
        ax4.plot(range(len(agv4error)), agv4error)
        ax5.plot(range(len(agv5error)), agv5error)
        plt.show()
        cv2.waitKey(0)
