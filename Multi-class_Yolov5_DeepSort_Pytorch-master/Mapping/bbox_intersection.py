import cv2
import numpy as np
from Mapping.mapper import *
CLASSES = ['AGV', 'Human']
CAMERAS = ['','WN', 'MSW', 'NS', 'ME', 'MW', 'EN']
def draw_multiple_boxes(bbox_list, mapping_objects, identities_list, classes_list, cam_id_list, offset=(0,0)):
    # bbox_list = list with bounding boxes from cameras
    # mapping_objects = Homography from the different cameras
    mappedImg = cv2.imread('Mapping/plane.png') #PLANAR_MAP
    classes_list = [item.tolist() for item in classes_list]
    #print('List in function: ', classes_list)
    j = 0
    for bbox, mapperObject, identities, cls, cam_id in zip(bbox_list, mapping_objects, identities_list, classes_list,cam_id_list):
        for (i, box), id in zip(enumerate(bbox), cam_id):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            color = compute_color_for_labels(id)

            # Mapping to plane
            x1m, x2m, y1m, y2m, color = mapperObject.mapBoundingBoxPoints(x1, x2, y1, y2, color)

            cv2.rectangle(mappedImg, (x1m, y1m), (x2m, y2m), color, 2)

            label = CAMERAS[int(id)] #CLASSES[cls[i]]


            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(mappedImg, (x1m, y1m), (x1m + t_size[0] + 3, y1m + t_size[1] + 4), color, -1)
            cv2.putText(mappedImg, label, (x1m, y1m + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return mappedImg

def draw_bboxes(bboxlist, mappedImg):
    #mappedImg = cv2.imread('Mapping/plane.png')
    for bbox in bboxlist:
        #print(bbox)
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        color = (166,151,79)
        cv2.rectangle(mappedImg, (x1, y1), (x2, y2), color, 2)
    return mappedImg




def iou_bboxes(bbox_list, mapObjects_list, cam_id_list, classes_list):
    bbox_all_list = np.array([[1,2,3,4,5]], dtype=int)
    # Collect all bboxes
    for (bbox, mapObject, bbox_class, cam_id) in zip(bbox_list, mapObjects_list, classes_list, cam_id_list):
        if bbox != []:
            for (i, box), camera_id in zip(enumerate(bbox), cam_id):
                camera_id = int(camera_id)
                x1c, y1c, x2c, y2c = [int(i) for i in box]


                p1 = np.array([[x1c, y1c]], dtype='float32')
                p2 = np.array([[x2c, y2c]], dtype='float32')

                p1 = np.array([p1])
                p2 = np.array([p2])

                x1, y1 = mapObject.getPoint(p1)
                x2, y2 = mapObject.getPoint(p2)

                bbox_w = abs(x2 - x1)
                bbox_h = abs(y2 - y1)

                # Change top left corner and bottom right corner depending on which direction the camera is pointing
                # to make it correct against the mapped view
                if camera_id == 2 or camera_id == 5: # Camera MW and MSW
                    bbox_all_list = np.append(bbox_all_list, [[x1, y2, bbox_h, bbox_w, bbox_class]], axis=0)
                elif camera_id == 3: # Camera NS
                    bbox_all_list = np.append(bbox_all_list, [[x2, y2, bbox_h, bbox_w, bbox_class]], axis=0)
                elif camera_id == 4: # Camera ME
                    bbox_all_list = np.append(bbox_all_list, [[x2, y1, bbox_h, bbox_w, bbox_class]], axis=0)
                else:
                    bbox_all_list = np.append(bbox_all_list, [[x1, y1, bbox_h, bbox_w, bbox_class]], axis=0)

    bbox_all_list = np.delete(bbox_all_list, (0), axis=0)
    candidates = np.copy(bbox_all_list)

    matches = np.array([[1, 2, 3, 4]], dtype=int)
    deleted_bbox = []

    for bbox in bbox_all_list:
        bbox_cam_ID = cam_id_list.pop(0)
        bbox_class = classes_list.pop(0)
        GO = True
        for box in deleted_bbox:
            if all(bbox == box):
                GO = False

        if GO:
            if candidates.shape[0] > 1:
                candidates = np.delete(candidates, 0, axis=0)
                if bbox_class == 1:
                    print('Human')
                intersected_bbox, matched_bbox_index = intersection(bbox, bbox_cam_ID, bbox_class, candidates,
                                                                    cam_id_list)

                if matched_bbox_index is not None:
                    deleted_bbox.append(candidates[matched_bbox_index, :].tolist())
                    candidates = np.delete(candidates, matched_bbox_index, axis=0) # Remove matched candidate

                matches = np.append(matches, [intersected_bbox], axis=0)

            else: # If no bounding boxes match, save the bbox as a "solo bbox" (i.e. an object seen only by one camera)
                intersected_bbox = [bbox[0], bbox[1], bbox[0]+bbox[3], bbox[1]+bbox[2]]

                matches = np.append(matches, [intersected_bbox], axis=0)
    matches = np.delete(matches, (0), axis=0)

    return matches

def intersection(bbox, bbox_cam_ID, bbox_class, candidates, cam_id_list):
    bb1x1 = bbox[0]
    bb1y1 = bbox[1]
    bb1x2 = bbox[0] + bbox[3]
    bb1y2 = bbox[1] + bbox[2]
    bbox_class = bbox[4]

    # determine the coordinates of the intersection rectangle
    iouList = []
    intersection_bbox = []

    for (i, candidate), Cam_ID in zip(enumerate(candidates), cam_id_list):
        cls = candidate[4]
        if (Cam_ID is not bbox_cam_ID) and (cls == bbox_class):

            bb2x1 = candidate[0]
            bb2y1 = candidate[1]
            bb2x2 = candidate[0] + candidate[3]
            bb2y2 = candidate[1] + candidate[2]

            x_left = max(bb1x1, bb2x1)
            y_top = max(bb1y1, bb2y1)
            x_right = min(bb1x2, bb2x2)
            y_bottom = min(bb1y2, bb2y2)


            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (bb1x2 - bb1x1) * (bb1y2 - bb1y1)
            bb2_area = (bb2x2 - bb2x1) * (bb2y2 - bb2y1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            if x_right < x_left or y_bottom < y_top:
                iouList.append(0.0)
            else:
                iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
                iouList.append(iou)
                if any(iou > item for item in iouList):
                    bbox_h = y_bottom - y_top
                    bbox_w = x_right - x_left
                    intersection_bbox = [x_left, y_top, x_right, y_bottom]
                    index = i

    #if len(iouList > 0) > 1:


    if intersection_bbox == []:
        intersection_bbox = [bb1x1, bb1y1, bb1x2, bb1y2]
        index = None

    return intersection_bbox, index

def compute_iou_matrix(bbox_list, mapObjects_list):
    bbox_list = bbox_to_coords(bbox_list, mapObjects_list)
    iou_matrix = np.zeros((len(bbox_list), len(bbox_list)))

    for i, bbox in enumerate(bbox_list):
        bb1x1 = bbox[0]
        bb1y1 = bbox[1]
        bb1x2 = bbox[0] + bbox[3]
        bb1y2 = bbox[1] + bbox[2]

        # determine the coordinates of the intersection rectangle
        iouList = []
        intersection_bbox = []
        for j, candidate in enumerate(bbox_list):
            bb2x1 = candidate[0]
            bb2y1 = candidate[1]
            bb2x2 = candidate[0] + candidate[3]
            bb2y2 = candidate[1] + candidate[2]

            x_left = max(bb1x1, bb2x1)
            y_top = max(bb1y1, bb2y1)
            x_right = min(bb1x2, bb2x2)
            y_bottom = min(bb1y2, bb2y2)


            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (bb1x2 - bb1x1) * (bb1y2 - bb1y1)
            bb2_area = (bb2x2 - bb2x1) * (bb2y2 - bb2y1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            if x_right < x_left or y_bottom < y_top:
                iou_matrix[i, j] = 0.0
            elif j == i:
                iou_matrix[i,j] = -1
            else:
                iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
                iou_matrix[i, j] = iou

    return iou_matrix




def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def bbox_to_coords(bbox_list, mapObjects_list):
    j = 0
    bbox_all_list = np.array([[1,2,3,4]], dtype='int')
    for (bbox, mapObject) in zip(bbox_list, mapObjects_list):
        if bbox != []:
            for i, box in enumerate(bbox):
                x1c, y1c, x2c, y2c = [int(i) for i in box]

                p1 = np.array([[x1c, y1c]], dtype='float32')
                p2 = np.array([[x2c, y2c]], dtype='float32')

                p1 = np.array([p1])
                p2 = np.array([p2])

                x1, y1 = mapObject.getPoint(p1)
                x2, y2 = mapObject.getPoint(p2)

                bbox_w = abs(x2 - x1)
                bbox_h = abs(y2 - y1)

                # Change top left corner and bottom right corner depending on which direction the camera is pointing
                # to make it correct against the mapped view
                if j == 4 or j == 1:  # Camera MW and MSW
                    bbox_all_list = np.append(bbox_all_list, [[x1, y2, bbox_h, bbox_w]], axis=0)
                elif j == 2:  # Camera NS
                    bbox_all_list = np.append(bbox_all_list, [[x2, y2, bbox_h, bbox_w]], axis=0)
                elif j == 3:  # Camera ME
                    bbox_all_list = np.append(bbox_all_list, [[x2, y1, bbox_h, bbox_w]], axis=0)
                else:
                    bbox_all_list = np.append(bbox_all_list, [[x1, y1, bbox_h, bbox_w]], axis=0)
        j += 1

    bbox_all_list = np.delete(bbox_all_list, 0,  axis=0)
    return bbox_all_list

def find_intersections(bbox_list, mapObjects_list, classes_list):
    iou_matrix = compute_iou_matrix(bbox_list, mapObjects_list)
    available_bboxes = [i for i in range(iou_matrix.shape[0])]
    intersecting_bboxes = []
    for index in range(iou_matrix.shape[0]):
        intersected_bbox = []

        if index in available_bboxes:
            intersected_bbox.append(index)
            available_bboxes.remove(index)
            class_bbox = classes_list[index]

            end = False
            switcher = 0  # Switch between row/column search

            while not end:
                currentIndex = index

                if switcher == 0:
                    #print(iou_matrix[currentIndex,:])
                    loop_flag = True
                    if np.max(iou_matrix[currentIndex,:]) == 0:
                        end = True
                        loop_flag = False  # Bounding box does not have any overlapping matches

                    i = 1
                    while loop_flag:  # Search for a candidate with the most iou overlapping
                        iou_value = np.partition(iou_matrix[currentIndex,:].flatten(), -i)[-i]

                        index = np.where(iou_matrix[currentIndex,:] == iou_value)[-1]
                        if iou_value == 0:
                            loop_flag = False
                            end = True

                        if len(index) > 1:
                            loop_flag = False
                            end = True
                            break
                        else:
                            if index in available_bboxes:
                                if all([iou_matrix[i,index] > 0 for i in intersected_bbox]) and (class_bbox == classes_list[index[0]]):
                                    intersected_bbox.append(int(index))
                                    available_bboxes.remove(index)
                                    loop_flag = False
                                else:
                                    i += 1
                            else:
                                i += 1

                    switcher = 1

                elif switcher == 1:
                    #print(iou_matrix[:, index])
                    currentIndex = index
                    i = 1
                    loop_flag = True
                    while loop_flag:  # Search for a candidate with the most iou overlapping
                        iou_value = np.partition(iou_matrix[:, currentIndex].flatten(), -i)[-i]

                        if iou_value == 0:
                            loop_flag = False
                            end = True
                        elif (iou_value == 0) and (i == 1):
                            loop_flag = False
                            end = True
                        else:
                            index = np.where(iou_matrix[currentIndex, :].flatten() == iou_value)[-1]
                            if len(index) > 1:
                                loop_flag = False
                                break
                            else:
                                if index in available_bboxes:
                                    if all([iou_matrix[index, i] > 0 for i in intersected_bbox]) and (class_bbox == classes_list[index[0]]):
                                        intersected_bbox.append(int(index))
                                        available_bboxes.remove(index)

                                        loop_flag = False
                                        switcher = 0
                                    else:
                                        i += 1
                                else:
                                    i += 1



            """else:
            end = True
            break"""
            #print(index)

            intersecting_bboxes.append(intersected_bbox)

    return intersecting_bboxes
def compute_multiple_intersection_bboxes(intersecting_bboxes, bbox_listed):
    intersected_bboxes = []
    bboxes_xyah = []
    bbox_list = []
    for bbox in bbox_listed:
        if bbox != []:
            bbox_list.append(bbox.tolist())
    for bbox_indexes in intersecting_bboxes:
        if len(bbox_indexes) > 1:
            bbx1 = bbox_list[bbox_indexes[0]]
            bb1x1 = bbx1[0]
            bb1y1 = bbx1[1]
            bb1x2 = bbx1[0] + bbx1[3]
            bb1y2 = bbx1[1] + bbx1[2]

            bbx2 = bbox_list[bbox_indexes[1]]
            bb2x1 = bbx2[0]
            bb2y1 = bbx2[1]
            bb2x2 = bbx2[0] + bbx2[3]
            bb2y2 = bbx2[1] + bbx2[2]

            x_left = max(bb1x1, bb2x1)
            y_top = max(bb1y1, bb2y1)
            x_right = min(bb1x2, bb2x2)
            y_bottom = min(bb1y2, bb2y2)

            bbxi = [x_left, y_top, x_right, y_bottom]
            bbx_xyah = [x_left, y_top, abs(x_left - x_right) / abs(y_top - y_bottom), abs(y_top - y_bottom)]

            if len(bbox_indexes) > 2:
                for i in range(2,len(bbox_indexes)):
                    bbxNext = bbox_list[bbox_indexes[i]]
                    bbNextx1 = bbxNext[0]
                    bbNexty1 = bbxNext[1]
                    bbNextx2 = bbxNext[0] + bbxNext[3]
                    bbNexty2 = bbxNext[1] + bbxNext[2]
                    bbox_class = bbxNext[4]

                    x_left = max(bbxi[0], bbNextx1)
                    y_top = max(bbxi[1], bbNexty1)
                    x_right = min(bbxi[2], bbNextx2)
                    y_bottom = min(bbxi[3], bbNexty2)

                    bbxi = [x_left, y_top, x_right, y_bottom]
                    bbx_xyah = [x_left, y_top, abs(x_left - x_right) / abs(y_top - y_bottom), abs(y_top - y_bottom)]

                intersected_bboxes.append(bbxi)
                bboxes_xyah.append(bbx_xyah)
            else:
                intersected_bboxes.append(bbxi)
                bboxes_xyah.append(bbx_xyah)
        else:
            bbx1 = bbox_list[bbox_indexes[0]]
            bb1x1 = bbx1[0]
            bb1y1 = bbx1[1]
            bb1x2 = bbx1[0] + bbx1[3]
            bb1y2 = bbx1[1] + bbx1[2]

            bbxi = [bb1x1, bb1y1, bb1x2, bb1y2]
            bbx_xyah = [x_left, y_top, abs(x_left - x_right) / abs(y_top - y_bottom), abs(y_top - y_bottom)]
            intersected_bboxes.append(bbxi)
            bboxes_xyah.append(bbx_xyah)

    return intersected_bboxes, bboxes_xyah


    # determine the coordinates of the intersection rectangle
    iouList = []
    intersection_bbox = []

    for (i, candidate), Cam_ID in zip(enumerate(candidates), cam_id_list):
        cls = candidate[4]
        if (Cam_ID is not bbox_cam_ID) and (cls == bbox_class):

            bb2x1 = candidate[0]
            bb2y1 = candidate[1]
            bb2x2 = candidate[0] + candidate[3]
            bb2y2 = candidate[1] + candidate[2]

            x_left = max(bb1x1, bb2x1)
            y_top = max(bb1y1, bb2y1)
            x_right = min(bb1x2, bb2x2)
            y_bottom = min(bb1y2, bb2y2)

            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (bb1x2 - bb1x1) * (bb1y2 - bb1y1)
            bb2_area = (bb2x2 - bb2x1) * (bb2y2 - bb2y1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            if x_right < x_left or y_bottom < y_top:
                iouList.append(0.0)
            else:
                iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
                iouList.append(iou)
                if any(iou > item for item in iouList):
                    bbox_h = y_bottom - y_top
                    bbox_w = x_right - x_left
                    intersection_bbox = [x_left, y_top, x_right, y_bottom]
                    index = i


def map_bboxes(bbox_list, mapObjects_list, cam_id_list, classes_list):
    bbox_all_list = np.array([[1,2,3,4,5]], dtype=int)
    # Collect all bboxes
    for (bbox, mapObject, bbox_class, cam_id) in zip(bbox_list, mapObjects_list, classes_list, cam_id_list):
        if bbox != []:
            for (i, box), camera_id in zip(enumerate(bbox),cam_id):
                camera_id = int(camera_id)
                x1c, y1c, x2c, y2c = [int(i) for i in box]


                p1 = np.array([[x1c, y1c]], dtype='float32')
                p2 = np.array([[x2c, y2c]], dtype='float32')

                p1 = np.array([p1])
                p2 = np.array([p2])

                x1, y1 = mapObject.getPoint(p1)
                x2, y2 = mapObject.getPoint(p2)

                bbox_w = abs(x2 - x1)
                bbox_h = abs(y2 - y1)

                # Change top left corner and bottom right corner depending on which direction the camera is pointing
                # to make it correct against the mapped view
                if camera_id == 5 or camera_id == 2: # Camera MW and MSW
                    bbox_all_list = np.append(bbox_all_list, [[x1, y2, bbox_h, bbox_w, bbox_class]], axis=0)
                elif camera_id == 3: # Camera NS
                    bbox_all_list = np.append(bbox_all_list, [[x2, y2, bbox_h, bbox_w, bbox_class]], axis=0)
                elif camera_id == 4: # Camera ME
                    bbox_all_list = np.append(bbox_all_list, [[x2, y1, bbox_h, bbox_w, bbox_class]], axis=0)
                else:
                    bbox_all_list = np.append(bbox_all_list, [[x1, y1, bbox_h, bbox_w, bbox_class]], axis=0)

    bbox_all_list = np.delete(bbox_all_list, (0), axis=0)
    return bbox_all_list
"""list = [[1,2],[3,4]]
print(list.pop(0))
print(list)"""

iou_matrix =  np.array([[-1,  0      ,     0  ,         0   ,        0         ,  0          , 0         ,  0    , 0.20826      ,     0       ,    0        ,   0     ,      0       ,    0     ,      0],
                        [0, -1     ,      0        ,   0  ,         0       ,    0          , 0   ,        0      ,     0,
                         0.16697     ,      0    ,       0   ,        0     ,      0       ,    0],
                        [0   ,        0 ,-1       ,    0     ,      0      ,     0      ,     0    ,       0     ,      0,
                         0       ,    0    ,       0    ,       0   ,        0      ,     0],
                        [0    ,       0         ,  0 ,-1,           0   ,        0        ,   0      ,     0       ,    0,
                         0      ,     0  ,  0.011692      ,     0    ,       0     ,      0],
                        [0  ,         0    ,       0,           0 ,-1,     0.11476      ,     0  ,         0       ,    0,
                         0         ,  0    ,       0  ,   0.16694  ,   0.21495     ,      0],
                        [0       ,    0         ,  0     ,      0    , 0.11476, -1,           0      ,     0       ,    0,
                         0         ,  0   ,        0     ,0.24507   ,  0.41431    ,       0],
                        [0    ,       0  ,         0        ,   0       ,    0     ,      0, -1,           0    ,       0,
                         0  ,   0.29808   ,        0           ,0      ,     0      ,     0],
                        [0         ,  0      ,     0   ,        0,           0      ,     0    ,       0, -1,           0,
                         0     ,      0      ,     0     ,      0    ,       0  ,   0.31299],
                        [0.20826   ,        0     ,      0      ,     0      ,     0          , 0    ,       0    ,       0, -1,
                         0     ,      0        ,   0     ,      0      ,     0       ,    0],
                        [0  ,   0.16697    ,       0 ,          0           ,0    ,       0   ,        0        ,   0,
                         0, -1,           0     ,      0      ,     0  ,         0  ,         0],
                        [0   ,        0    ,       0 ,          0  ,         0         ,  0   ,  0.29808  ,         0,
                         0         ,  0 ,-1,           0         ,  0        ,   0     ,      0],
                        [0        ,   0    ,       0  ,  0.011692 ,          0        ,   0       ,    0       ,    0,
                         0        ,   0       ,    0, -1,           0    ,       0     ,      0],
                        [0     ,      0      ,     0     ,      0,     0.16694  ,   0.24507   ,        0  ,         0,
                         0    ,       0    ,       0   ,        0, -1,     0.12194     ,      0],
                        [0    ,       0     ,      0      ,     0   ,  0.21495  ,   0.41431   ,        0     ,      0,
                         0      ,     0     ,      0     ,      0 ,    0.12194, -1,           0],
                        [0   ,        0    ,       0      ,     0     ,      0    ,       0    ,       0 ,    0.31299,
                         0   ,        0         ,  0     ,      0      ,     0          , 0 ,-1]])
#print(iou_matrix)
#print(find_intersections(iou_matrix))
