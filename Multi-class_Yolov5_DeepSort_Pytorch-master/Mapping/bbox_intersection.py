import cv2
import numpy as np
from deep_sort.deep_sort.sort.iou_matching import *
from Mapping.mapper import *

def draw_multiple_boxes(bbox_list, mapping_objects, identities_list, offset=(0,0)):
    # bbox_list = list with bounding boxes from cameras
    # mapping_objects = Homography from the different cameras
    mappedImg = cv2.imread('Mapping/plane.png') #PLANAR_MAP
    for bbox, mapperObject, identities in zip(bbox_list, mapping_objects, identities_list):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            id = int(identities[i]) if identities is not None else 0
            color = compute_color_for_labels(id)

            # Mapping to plane
            x1m, x2m, y1m, y2m, color = mapperObject.mapBoundingBoxPoints(x1, x2, y1, y2, color)
            cv2.rectangle(mappedImg, (x1m, y1m), (x2m, y2m), color, 2)
            #cv2.line(mappedImg, (x1m, y1m), (x1m, y2m), color, 2)  # left line
            #cv2.line(mappedImg, (x2m, y1m), (x2m, y2m), color, 2)  # right line
            #cv2.line(mappedImg, (x1m, y1m), (x2m, y1m), color, 2)  # top line
            #cv2.line(mappedImg, (x1m, y2m), (x2m, y2m), color, 2)  # bottom line
            # cv2.rectangle(mappedImg, (x1m, y1m), (x2m, y2m), color, 1)
            # allMappedPoints.append(mappedPoint)

    return mappedImg

def draw_bboxes(bboxlist):
    mappedImg = cv2.imread('Mapping/plane.png')
    for bbox in bboxlist:
        print(bbox)
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        color = (166,151,79)
        cv2.rectangle(mappedImg, (x1, y1), (x2, y2), color, 2)
    return mappedImg

def iou_bboxes(bbox_list, mapObjects_list):
    bbox_all_list = np.array([[1,2,3,4]], dtype=int)
    # Collect all bboxes
    j = 0
    for (bbox, mapObject) in zip(bbox_list, mapObjects_list):
        for i, box in enumerate(bbox):
            x1c, y1c, x2c, y2c = [int(i) for i in box]


            p1 = np.array([[x1c, y1c]], dtype='float32')
            p2 = np.array([[x2c, y2c]], dtype='float32')

            p1 = np.array([p1])
            p2 = np.array([p2])

            x1, y1 = mapObject.getPoint(p1)
            x2, y2 = mapObject.getPoint(p2)
            print('x1y1x2y2: ', x1,y1,x2,y2)
            bbox_w = abs(x2 - x1)
            bbox_h = abs(y2 - y1)

            # Change top left corner and bottom right corner depending on which directon the camera is pointing to make it correct against the mapped view
            if j == 4 or j == 1: # Camera MW and MSW
                bbox_all_list = np.append(bbox_all_list, [[x1, y2, bbox_h, bbox_w]], axis=0)
            elif j == 2: # Camera NS
                bbox_all_list = np.append(bbox_all_list, [[x2, y2, bbox_h, bbox_w]], axis=0)
            elif j == 3: # Camera ME
                bbox_all_list = np.append(bbox_all_list, [[x2, y1, bbox_h, bbox_w]], axis=0)
            else:
                bbox_all_list = np.append(bbox_all_list, [[x1, y1, bbox_h, bbox_w]], axis=0)
        j += 1

    bbox_all_list = np.delete(bbox_all_list, (0), axis=0)
    candidates = np.copy(bbox_all_list)

    matches = np.array([[1, 2, 3, 4]], dtype=int)
    deleted_bbox = []

    for bbox in bbox_all_list:

        print('Active bbox: ', bbox)
        print('deleted_bbox', deleted_bbox)
       # if [bbox] not in deleted_bbox:
        GO = True
        for box in deleted_bbox:
            if all(bbox == box):
                GO = False

        if GO:
            if candidates.shape[0] > 1:
                candidates = np.delete(candidates, 0, axis=0)
                print('candidates', candidates)
                intersected_bbox, matched_bbox_index = intersection(bbox, candidates)
                print('matched_bbox_index', matched_bbox_index)

                if matched_bbox_index is not None:
                    print('Matched bbox: ', candidates[matched_bbox_index, :])
                    deleted_bbox.append(candidates[matched_bbox_index, :].tolist())
                    candidates = np.delete(candidates, matched_bbox_index, axis=0)

                matches = np.append(matches, [intersected_bbox], axis=0)

                print('matched bbox = ', intersected_bbox)
            else: # If no bounding boxes match. Save the current one
                intersected_bbox = [bbox[0], bbox[1], bbox[0]+bbox[3], bbox[1]+bbox[2]]
                print('NONE ----->: ', intersected_bbox)
                if cv2.waitKey(0) == 33:
                    continue

                matches = np.append(matches, [intersected_bbox], axis=0)
        else:
            print('====================================')

    matches = np.delete(matches, (0), axis=0)

    return matches



def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def intersection(bbox, candidates):
    bb1x1 = bbox[0]
    bb1y1 = bbox[1]
    bb1x2 = bbox[0] + bbox[3]
    bb1y2 = bbox[1] + bbox[2]
    area_of_bbox = bbox[2] * bbox[3]

    # determine the coordinates of the intersection rectangle
    iouList = []
    intersection_bbox = []
    for i, candidate in enumerate(candidates):
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

    if intersection_bbox == []:
        intersection_bbox = [bb1x1, bb1y1, bb1x2, bb1y2]
        index = None

    return intersection_bbox, index

bbox = [204 ,1002  ,30  ,24]
candidate = [[  77  ,697 ,  58 ,  73],
[ 490, 1025  , 43  , 71],
 [1702 , 162 ,  57 ,  49],
 [1719 , 568   ,71  , 74],
 [1792  ,502   ,26 , 147],
 [1061  ,412  , 13  , 40],
 [  -4  ,770  , 92   ,88],
 [ -15  ,698  , 82,  117],
 [ 151  ,829 ,  87 , 128],
 [1403  ,445,   84  , 23]]
print(intersection(bbox, candidate))