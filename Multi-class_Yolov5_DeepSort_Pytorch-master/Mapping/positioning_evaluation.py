
from groundTruth import getGroundTruth, getGroundTruthEvaluation
import math

""" 
Part of Master Thesis 'Indoor Tracking using a Central Camera System' at Chalmers University of Technology, conducted
at Sigma Technology Insights 2021.

Authors:
Jonas Lindberg
Sara Roth

"""

# Convert pixels to meters
def pixel_to_meter(state):
    return [state[0]*50/1788, (1069 - state[1]) * 30/1069]

# Get the true position
AGV1Meter, AGV2Meter, AGV3Meter, AGV4Meter, AGV5Meter = getGroundTruthEvaluation()
true_position_list = [AGV1Meter, AGV2Meter, AGV3Meter, AGV4Meter, AGV5Meter]

#true_position_list = [AGV2Meter, AGV1Meter, Human1, AGV4Meter, AGV3Meter, Human2]
names_list = ['AGV1Meter', 'AGV2Meter', 'AGV3Meter', 'AGV4Meter', 'AGV5Meter']
#names_list = ['AGV2Meter', 'AGV1Meter', 'Human1', 'AGV4Meter', 'AGV3Meter', 'Human2']


# Get the filtered position
def filtered_positions(states, frame):
    """
    Get the filtered positions and compare with the true values to get the errors
    """
    for state, truth, names in zip(states, true_position_list, names_list):

        filter_x = pixel_to_meter(state)[0]
        filter_y = pixel_to_meter(state)[1]
        true_x = truth[frame-1][0]
        true_y = truth[frame-1][1]
        total_error_x = abs(filter_x - true_x)
        total_error_y = abs(filter_y - true_y)

        print(names)
        print('Filtered ', pixel_to_meter(state))
        print('True', truth[frame-1])
        print('Error x', total_error_x )
        print('Error y', total_error_y)
        print('Total error', math.sqrt(total_error_x**2 + total_error_y**2))
        print('-----------------------------------------')
