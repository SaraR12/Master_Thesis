# Optical Flow

import math

""" 
Part of Master Thesis 'Indoor Tracking using a Central Camera System' at Chalmers University of Technology, conducted
at Sigma Technology Insights 2021.

Authors:
Jonas Lindberg
Sara Roth

"""

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

    """ Create Optical flow objects and save states for the objects """
    
    def __init__(self, frame, id_b, center_point, class_name):
        self.frame = 0
        self.last_frame = 0
        self.id_b = id_b
        self.last_center_point = center_point
        self.this_center_point = center_point
        self.class_name = class_name
        self.opt_dict = {} 

    def __call__(self, frame, id_b, center_point, class_name, writeDict=False):

        # Update parameters
        self.id_b = id_b # update ID
        '''self.last_frame = f_curr # Update last frame to the previous current frame'''
        self.frame = frame # Update current frame to input frame
        '''self.last_BBOX = bbox_curr # Update last bbox to the previous current bbox'''

        self.this_center_point = center_point # Update to new center point
        self.class_name = class_name # Update class_name

        # Add new info to dictonary
        self.opt_dict[str(self.id_b)] = [self.id_b, self.last_frame, self.frame, self.last_center_point, self.this_center_point, self.class_name ]

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

        state = [center_point_change_xy, heading_xy]
        # state = [frame, id, camera ID = -1, class, np.array(prev_x prev_y), np.array(x, y), np.array(dx, dy)]

        self.last_frame = frame
        self.last_center_point = center_point

        return state

