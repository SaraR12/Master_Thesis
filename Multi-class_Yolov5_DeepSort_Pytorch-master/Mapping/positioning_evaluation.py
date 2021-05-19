from groundTruth import *
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Convert pixels to meters
def pixel_to_meter(state):
    return [state[0]*50/1080, (1069 - state[1]) * 30/1069]
def pixel_to_meterEval(state):
    return [state[0]*26/1080, (1080 - state[1]) * 26/1080]

def meter_to_pixel(state):
    return [state[0]* 1080/26, (26 - state[1]) * 1080/26]

# Get the true position
"""AGV1Meter, AGV2Meter, AGV3Meter, AGV4Meter, AGV5Meter = getGroundTruthEvaluation()
true_position_list = [AGV4Meter, AGV2Meter, AGV1Meter, AGV3Meter, AGV5Meter]"""
AGV1Meter = getGroundTruthOneAGV()
#AGV1Meter, AGV2Meter, AGV3Meter, AGV4Meter, AGV5Meter = getGroundTruthEvaluation()

#true_position_list = [AGV2Meter, AGV4Meter, AGV3Meter, AGV1Meter, AGV5Meter]
true_position_list = [AGV1Meter]
agv1error, agv2error, agv3error, agv4error, agv5error = [], [], [], [], []
errors = [agv1error, agv2error, agv3error, agv4error, agv5error]
errors = [agv1error]
errorx = []
errory = []
names_list = ['AGV1Meter', 'AGV2Meter', 'AGV3Meter', 'AGV4Meter', 'AGV5Meter']

#names_list = ['AGV1Meter']
# Get the filtered position
combined_total_error = []
state_list = []
state_x = []
state_y = []
true_state_x = []
true_state_y = []

estimated_velocity = []
true_velocity = []


plane = cv2.imread('videos/OneAGV/mapSmall.png')
def filtered_positions(states, frame):
    for state, truth, names, error in zip(states, true_position_list, names_list, errors):

        state_list.append(state)
        filter_x = pixel_to_meterEval(state)[0]
        filter_y = pixel_to_meterEval(state)[1]
        true_x = truth[frame-1][0]
        true_y = truth[frame-1][1]
        total_error_x = (filter_x - true_x)
        total_error_y = (filter_y - true_y)
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
    if frame == 298:
        print('MEDIAN ', np.median(combined_total_error))
        #print('Mean error AGV1 = ', np.mean(agv1error))
        """print('Mean error AGV2 = ', np.mean(agv2error))
        print('Mean error AGV3 = ', np.mean(agv3error))
        print('Mean error AGV4 = ', np.mean(agv4error))
        print('Mean error AGV5 = ', np.mean(agv5error))"""


        """fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        fig.suptitle('Positioning Error (meter)')
        ax1.plot(range(len(agv1error)), agv1error, label='Error')
        ax1.plot(range(len(agv1error)), np.ones(len(agv1error)) * np.mean(combined_total_error), label='Mean Error')


        ax2.plot(range(len(errorx)), errorx, label='Error')
        ax2.plot(range(len(errorx)), np.ones(len(errorx)) * np.mean(errorx), label='Mean Error')

        ax3.plot(range(len(errory)), errory, label='Error')
        ax3.plot(range(len(errory)), np.ones(len(errory)) * np.mean(errory), label='Mean Error')"""


        """ax1.legend()
        ax2.legend()
        ax3.legend()

        ax1.set_title('Total error [m]')
        ax2.set_title('Error X [m]')
        ax3.set_title('Error Y [m]')

        ax1.set_ylabel('Meter')
        ax2.set_ylabel('Meter')
        ax3.set_ylabel('Meter')

        ax1.set_xlabel('Frame')
        ax2.set_xlabel('Frame')
        ax3.set_xlabel('Frame')

        #ax4.plot(range(len(agv4error)), agv4error)
        #ax5.plot(range(len(agv5error)), agv5error)
        plt.show()
        cv2.waitKey(0)"""



        #fig, (ax1, ax2) = plt.subplots(1,2)
        state_old = pixel_to_meterEval(state_list[0])
        true_state_old = true_position_list[0][0]
        i = 0
        for state, true_state in zip(state_list, true_position_list[0]):
            if i == 60:
                print('hej')
            state = pixel_to_meterEval(state)


            state_x.append(state[0])
            state_y.append(state[1])
            true_state_x.append(true_state[0])
            true_state_y.append(true_state[1])

            #cv2.circle(plane, (int(true_state[0]), int(true_state[1])), 2, (0, 0, 0), 2)


            estimated_velocity.append(math.sqrt((state[0] - state_old[0])**2 + (state[1] - state_old[1])**2)/(1/24))
            true_velocity.append(math.sqrt((true_state[0] - true_state_old[0])**2 + (true_state[1] - true_state_old[1])**2) / (1 / 24))

            state_old = state
            true_state_old = true_state
            i += 1

        """cv2.imshow('namn', plane)
        cv2.waitKey(0)"""
        """state = pixel_to_meterEval(state)
        state_x.append(state[0])
        state_y.append(state[1])
        true_state_x.append(true_state[0])
        true_state_y.append(true_state[1])"""


        """fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.plot(range(len(state_x)), state_x, label='Estimated X')
        ax1.plot(range(len(true_state_x)), true_state_x, label='True X')

        ax2.plot(range(len(state_y)), state_y, label='Estimated Y')
        ax2.plot(range(len(true_state_y)), true_state_y, label='True Y')

        ax1.set_title('Comparison position in X [m]')
        ax2.set_title('Comparison position in Y [m]')


        ax1.set_ylabel('Meter')
        ax2.set_ylabel('Meter')
        ax1.set_xlabel('Frame')
        ax2.set_xlabel('Frame')

        ax1.legend(), ax2.legend()"""

        plt.plot(range(len(true_velocity)), true_velocity, label='True Velocity')

        plt.plot(range(len(estimated_velocity)), estimated_velocity, label='Estimated Velocity')
        plt.plot(range(len(estimated_velocity)), np.ones(len(estimated_velocity))*0.4519921169328904, label='Mean Velocity Error')
        plt.legend()
        plt.title('Comparison between estimated and true velocity')
        plt.xlabel('Frame')
        plt.ylabel('Meter per Second')


        plt.show()
        print('hej')
            #cv2.circle(plane, (int(state[0]),int(state[1])), 2, (0, 0, 0), 2)


