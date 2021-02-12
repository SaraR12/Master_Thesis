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
    for (bbox, mapObject) in zip(bbox_list,mapObjects_list):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]

            p1 = np.array([[x1, y1]], dtype='float32')
            p2 = np.array([[x2, y2]], dtype='float32')

            p1 = np.array([p1])
            p2 = np.array([p2])

            x1, y1 = mapObject.getPoint(p1)
            x2, y2 = mapObject.getPoint(p2)
            bbox_w = abs(x2 - x1)
            bbox_h = abs(y2 - y1)
            bbox_all_list = np.append(bbox_all_list, [[x1, y1, bbox_h, bbox_w]], axis=0)

    bbox_all_list = np.delete(bbox_all_list, (0), axis=0)
    candidates = np.copy(bbox_all_list)

    matches = np.array([[1, 2, 3, 4]], dtype=int)
    for bbox in bbox_all_list:
        candidates = np.delete(candidates, (0), axis=0)

        #print('candidates',candidates)
        if candidates.shape[0] > 1:
            candidate_IOU = iou(bbox, candidates)
            matched_bbox = candidates[np.argmax(candidate_IOU), :]

            x1 = max(bbox[0], matched_bbox[0])
            y1 = min(bbox[1], matched_bbox[1])

            x2 = max((bbox[0] + bbox[3]), (matched_bbox[0] + matched_bbox[3]))
            y2 = min((bbox[1] - bbox[2]), (matched_bbox[1] - matched_bbox[2]))

            print('bbox= ', bbox)
            print('matched bbox = ', matched_bbox)
            print('x1,y1,x2,y2 = ', x1, y1, x2, y2)
            if cv2.waitKey(0) == 33:
                continue


            matches = np.append(matches, [[x1, y1, x2, y2]], axis=0)

    matches = np.delete(matches, (0), axis=0)

    return matches



def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)