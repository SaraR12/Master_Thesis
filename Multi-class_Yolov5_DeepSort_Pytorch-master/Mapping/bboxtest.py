import cv2
import numpy as np
from deep_sort.deep_sort.sort.iou_matching import *
"""plane = cv2.imread('plane.png')
cv2.rectangle(plane, (31,678), (107, 733), (255,0,0))
cv2.rectangle(plane, (32,746), (32+74,746+80),(0,255,0))
cv2.rectangle(plane, (32,768), (-42,666),(0,0,255))
cv2.imshow('1', plane)


cv2.waitKey(0)
"""
def iou_bboxes(bbox_list):
    bbox_all_list = np.array([[1,2,3,4]], dtype=int)
    # Collect all bboxes
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        bbox_w = abs(x2 - x1)
        bbox_h = abs(y2 - y1)
        bbox_all_list = np.append(bbox_all_list, [[x1, y1, bbox_h, bbox_w]], axis=0)

    bbox_all_list = np.delete(bbox_all_list, (0), axis=0)
    candidates = np.copy(bbox_all_list)

    matches = np.array([[1, 2, 3, 4]], dtype=int)

    for bbox in bbox_all_list:
        #candidates = np.delete(candidates, (0), axis=0)

        #print('candidates',candidates)
        if candidates.shape[0] > 1:
            candidate_IOU = iou(bbox, candidates)
            index = np.argmax(candidate_IOU)
            matched_bbox = candidates[index, :]

            # Inverted coordinate-system (top left is (0,0))
            x1 = max(bbox[0], matched_bbox[0])
            y1 = max(bbox[1], matched_bbox[1])

            x2 = min((bbox[0] + bbox[3]), (matched_bbox[0] + matched_bbox[3]))
            y2 = min((bbox[1] + bbox[2]), (matched_bbox[1] + matched_bbox[2]))

            print('bbox= ', bbox)
            print('matched bbox = ', matched_bbox)
            print('x1,y1,x2,y2 = ', x1, y1, x2, y2)
            if cv2.waitKey(0) == 33:
                continue


            matches = np.append(matches, [[x1, y1, x2, y2]], axis=0)

    matches = np.delete(matches, (0), axis=0)

    return matches

iou_bboxes([[0,50,50,0],[1,70,51,30], [30,50,50,50],[31,70,51,60]])