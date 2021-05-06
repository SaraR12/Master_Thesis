# Optical Flow

# RUN object_tracker.py
# conda activate yolov4-cpu
# python3 object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny

import cv2
import numpy as np
import math

"""Solve optical flow problem

    Parameters 
    ----------
    frame : int 
        Current frame number.
    id_b : int
        Current ID number of detected object.
    BBOX : List[top left x, top left y, width, height ]
        A list of current bounding box coordinates in tlwh (top left width height) format
    class_name : str
        A string of current class name

    Returns
    -------
    state = List(int, int, int, np.array([int, int]), np.array([int, int]), np.array([int, int]))
    state = List(frame, id, camera ID, previous BBOX center, current BBOX center, delta[x,y])
        Returns a list with the following six entries:
        * frame
        * id
        * camera ID: -1
        * previous BBOX center: x_last and y_last center position
        * current BBOX center: x_curr and y_curr center position
        * delta: dx and dy, change in position between detections, [pixel/frame]
    """ 

    
class OpticalFlow:
    
    def __init__(self, frame, id_b, center_point, class_name):
        self.frame = 0
        self.last_frame = 0
        self.id_b = id_b
        self.last_center_point = center_point
        self.this_center_point = center_point
        self.class_name = class_name
        self.opt_dict = {} 

    def __call__(self, frame, id_b, center_point, class_name, writeDict=False):
        
        ### REMOVE LATER ####
        if writeDict:
            PATH_DICT = './outputs/dict_test.txt'
            PATH_OUT = './outputs/optical_test.txt'

        # Fill dictonary
        '''if str(id_b) not in self.opt_dict: # If no stored info about current ID, add info.
            # Add first ID found
            self.opt_dict[str(id_b)] = [id_b, frame, frame, BBOX, BBOX, class_name]
            # Can not make calculations here since no info about ID is stored

            #print('Found new ID and added to dictionary')
        else:
            #print('Existing ID in dict')

            # Extract info from dictonary for ID
            ID, f_last, f_curr, bbox_last, bbox_curr, class_name = self.opt_dict[str(id_b)]
        '''
        # Update parameters
        self.id_b = id_b # update ID
        '''self.last_frame = f_curr # Update last frame to the previous current frame'''
        self.frame = frame # Update current frame to input frame
        '''self.last_BBOX = bbox_curr # Update last bbox to the previous current bbox'''

        self.this_center_point = center_point # Update to new center point
        self.class_name = class_name # Update class_name

        # Add new info to dictonary
        self.opt_dict[str(self.id_b)] = [self.id_b, self.last_frame, self.frame, self.last_center_point, self.this_center_point , self.class_name ]

        if writeDict:
            with open(PATH_DICT, 'a') as fs:
                fs.write(str(self.opt_dict[str(self.id_b)]) + "\n")

        #### COMPUTATIONS ####

        # Frame delta
        frame_delta = self.frame - self.last_frame

        # Box delta in pixels/frame, vector of delta x and delta y 
        if frame_delta == 0:
            center_point_change_xy = (self.this_center_point - self.last_center_point)
            heading_xy = math.atan(center_point_change_xy[1]/center_point_change_xy[0])
            heading_xy = math.degrees(heading_xy)
        else:
            try:
                center_point_change_xy = (self.this_center_point - self.last_center_point)/frame_delta
            except:
                print('failing')
            heading_xy = math.atan(center_point_change_xy[1] / center_point_change_xy[0])
            heading_xy = math.degrees(heading_xy)

            #state = [self.frame, self.id_b, -1, self.class_name, np.array([last_box_center_x, last_box_center_y]), np.array([this_box_center_x, this_box_center_y]), box_delta_xy]
        state = [center_point_change_xy, heading_xy]
        # state = [frame, id, camera ID = -1, class, np.array(prev_x prev_y), np.array(x, y), np.array(dx, dy)]

        self.last_frame = self.frame
        self.last_center_point = self.center_point

        ############REMOVE LATER ###########
        if writeDict:
            with open(PATH_OUT, 'a') as fs:
                fs.write(str(state) + "\n")
        ####################################

        return state

