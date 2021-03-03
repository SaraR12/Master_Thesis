import cv2
from Mapping.mapper import *
from shapely.geometry import Polygon


CLASSES = ['AGV', 'Human']
CAMERAS = ['','WN', 'MSW', 'NS', 'M', 'EN', 'ME', 'WN2', 'MW']

#### POLYGON AREA ####
def areaOfPolygon(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))
def segments(p):
    return zip(p, p[1:] + [p[0]])

def draw_multiple_boxes(bbox_list, mapping_objects, identities_list, classes_list, cam_id_list, img, offset=(0,0)):
    # bbox_list = list with bounding boxes from cameras
    # mapping_objects = Homography from the different cameras
    mappedImg = img
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
            pTL = np.array([[x1, y1]], dtype='float32')
            pTL = np.array([pTL])
            xTL, yTL = mapperObject.getPoint(pTL)

            pTR = np.array([[x2, y1]], dtype='float32')
            pTR = np.array([pTR])
            xTR, yTR = mapperObject.getPoint(pTR)

            pBL = np.array([[x1, y2]], dtype='float32')
            pBL = np.array([pBL])
            xBL, yBL = mapperObject.getPoint(pBL)

            pBR = np.array([[x2, y2]], dtype='float32')
            pBR = np.array([pBR])
            xBR, yBR = mapperObject.getPoint(pBR)

            pts = np.array([[xTL, yTL], [xBL, yBL],[xBR, yBR], [xTR, yTR]],
                           np.int32)

            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(mappedImg, [pts],
                          True, color, 2)

            label = str(CAMERAS[int(id)]) #CLASSES[cls[i]]
            j += 1

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(mappedImg, (xTR, yTR), (xTR + t_size[0] + 3, yTR + t_size[1] + 4), color, -1)
            cv2.putText(mappedImg, label, (xTR, yTR + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return mappedImg

def draw_bboxes(bboxlist, mappedImg):
    #mappedImg = cv2.imread('Mapping/plane.png')
    scale_x = 50 / 1788
    scale_y = 30 / 1069
    for bbox in bboxlist:
        center = bbox[-2]
        centerX = round(scale_x * center[0], 3)
        centerY = round((1069 - center[1]) * scale_y, 3)

        pts = np.array([])
        for i in range(len(bbox) -2):
            pts = np.append(pts, np.round(bbox[0]))
        #pts = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mappedImg, [np.int32(pts)], True, (255, 0, 0), 2,)
        cv2.circle(mappedImg, (np.int32(center[0]), np.int32(center[1])),2, (255,0,0),2)
        label = str('x = ' + str(centerX) + ' y = ' + str(centerY))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

        x = round(bbox[0][0])
        y = round(bbox[0][1])
        #cv2.rectangle(mappedImg, (x, y), (x + t_size[0] + 3, y + t_size[1] + 4), (255, 0, 0), -1)
        cv2.putText(mappedImg, label, (x, y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return mappedImg

"""def draw_bboxesKalman(bboxlist, mappedImg):
    #mappedImg = cv2.imread('Mapping/plane.png')
    for bbox in bboxlist:
        #print(bbox)
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        color = (166,151,79)
        cv2.rectangle(mappedImg, (x1, y1), (x2, y2), color, 2)
    return mappedImg"""


"""def iou_bboxes(bbox_list, mapObjects_list, cam_id_list, classes_list):
    bbox_all_list = np.array([[1,2,3,4,5]], dtype=int)
    # Collect all bboxes
    for (bbox, mapObject, bbox_class, cam_id) in zip(bbox_list, mapObjects_list, classes_list, cam_id_list):
        if bbox != []:
            for (i, box), id in zip(enumerate(bbox), cam_id):
                id = int(id)
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
                if id == 2 or id == 3:  # Camera MSW and NS
                    bbox_all_list = np.append(bbox_all_list, [[x1, y2, bbox_h, bbox_w, bbox_class]], axis=0)# bbox_all_list = np.append(bbox_all_list, [[y1, x2, bbox_w, bbox_h, bbox_class]], axis=0)
                elif id == 4 or id == 6:  # Camera M and ME
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

    return matches"""

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

def compute_iou_matrix(bbox_list, mapObjects_list, camera_id_list):
    bbox_list = bbox_to_coords(bbox_list, mapObjects_list, camera_id_list)
    iou_matrix = np.zeros((len(bbox_list), len(bbox_list)))

    for i, bbox in enumerate(bbox_list):
        bb1xTL = bbox[0]
        bb1yTL = bbox[1]
        bb1xTR = bbox[2]
        bb1yTR = bbox[3]
        bb1xBL = bbox[4]
        bb1yBL = bbox[5]
        bb1xBR = bbox[6]
        bb1yBR = bbox[7]
        p1 = Polygon([(bb1xTL, bb1yTL), (bb1xTR, bb1yTR), (bb1xBR, bb1yBR), (bb1xBL, bb1yBL)])

        """p11 = gpd.GeoSeries(p1)
        p11.plot()"""
        areaBbx1 = areaOfPolygon([[bb1xTL, bb1yTL], [bb1xTR, bb1yTR], [bb1xBR, bb1yBR], [bb1xBL, bb1yBL]])
        # determine the coordinates of the intersection rectangle
        iouList = []
        intersection_bbox = []
        for j, candidate in enumerate(bbox_list):
            bb2xTL = candidate[0]
            bb2yTL = candidate[1]
            bb2xTR = candidate[2]
            bb2yTR = candidate[3]
            bb2xBL = candidate[4]
            bb2yBL = candidate[5]
            bb2xBR = candidate[6]
            bb2yBR = candidate[7]
            p2 = Polygon([(bb2xTL, bb2yTL), (bb2xTR, bb2yTR), (bb2xBR, bb2yBR), (bb2xBL, bb2yBL)])
            """p22 = gpd.GeoSeries(p2)
            p22.plot()"""

            if p1.intersects(p2):
                intersection = p1.intersection(p2)
                """intersection1 = gpd.GeoSeries(intersection)
                intersection1.plot()
                plt.show()"""

                intersection = intersection.exterior.coords.xy
                intersected_bbox = []
                for x,y in zip(intersection[0], intersection[1]):
                    intersected_bbox.append([int(round(x)),int(round(y))])
                intersected_bbox.pop(-1)

                iou = areaOfPolygon(intersected_bbox) / areaBbx1
            else:
                iou = 0
            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box


            if iou == 1:
                iou_matrix[i,j] = -1
            else:
                iou_matrix[i, j] = iou

    return iou_matrix




def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def bbox_to_coords(bbox_list, mapObjects_list, camera_id_list):
    bbox_all_list = np.array([[1,2,3,4,5,6,7,8]], dtype='int')
    for (bbox, mapObject, camID) in zip(bbox_list, mapObjects_list, camera_id_list):
        if bbox != []:
            for (i, box), id in zip(enumerate(bbox), camID):
                id = int(id)

                x1c, y1c, x2c, y2c = [int(i) for i in box]

                pTL = np.array([[x1c,y1c]], dtype='float32')
                pTL = np.array([pTL])
                xTL, yTL = mapObject.getPoint(pTL)

                pTR = np.array([[x2c, y1c]], dtype='float32')
                pTR = np.array([pTR])
                xTR, yTR = mapObject.getPoint(pTR)

                pBL = np.array([[x1c, y2c]], dtype='float32')
                pBL = np.array([pBL])
                xBL, yBL = mapObject.getPoint(pBL)

                pBR = np.array([[x2c, y2c]], dtype='float32')
                pBR = np.array([pBR])
                xBR, yBR = mapObject.getPoint(pBR)


                # Change top left corner and bottom right corner depending on which direction the camera is pointing
                # to make it correct against the mapped view
                bbox_all_list = np.append(bbox_all_list, [[xTL, yTL, xTR, yTR, xBL, yBL, xBR, yBR]], axis=0)

    bbox_all_list = np.delete(bbox_all_list, 0,  axis=0)

    return bbox_all_list

def find_intersections(bbox_list, mapObjects_list, classes_list, camera_id_list):
    iou_matrix = compute_iou_matrix(bbox_list, mapObjects_list, camera_id_list)
    available_bboxes = [i for i in range(iou_matrix.shape[0])]
    intersecting_bboxes = []
    cls_list = []

    for index in range(iou_matrix.shape[0]):
        intersected_bbox = []

        if index in available_bboxes:
            intersected_bbox.append(index)
            available_bboxes.remove(index)
            class_bbox = classes_list[index]


            if class_bbox == 2:
                class_bbox = 0
            elif class_bbox == 3:
                class_bbox = 1

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
                        if i > 4:
                            print('WARNING1')

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
                                if classes_list[index[0]] == 2:
                                    classes_list[index[0]] = 0
                                elif classes_list[index[0]] == 3:
                                    classes_list[index[0]] = 1
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
                        if i > 4:
                            print('WARNING2')
                        if iou_value == 0:
                            loop_flag = False
                            end = True
                        elif (iou_value == 0) and (i == 1):
                            loop_flag = False
                            end = True
                        else:
                            index = np.where(iou_matrix[:, currentIndex].flatten() == iou_value)[-1]
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
            cls_list.append(class_bbox)
    return intersecting_bboxes, cls_list
def compute_multiple_intersection_bboxes(intersecting_bboxes, bbox_listed, class_list):
    intersected_bboxes = []
    measurements = []
    bbox_list = []

    for bbox in bbox_listed:
        if bbox != []:
            bbox_list.append(bbox.tolist())
    for bbox_indexes in intersecting_bboxes:
        cls = class_list[bbox_indexes[0]]

        if len(bbox_indexes) > 1:
            bbx1 = bbox_list[bbox_indexes[0]]
            bb1xTL = bbx1[0]
            bb1yTL = bbx1[1]
            bb1xTR = bbx1[2]
            bb1yTR = bbx1[3]
            bb1xBR = bbx1[4]
            bb1yBR = bbx1[5]
            bb1xBL = bbx1[6]
            bb1yBL = bbx1[7]

            b1 = Polygon([(bb1xTL, bb1yTL), (bb1xTR, bb1yTR), (bb1xBR, bb1yBR), (bb1xBL, bb1yBL)])

            bbx2 = bbox_list[bbox_indexes[1]]
            bb2xTL = bbx2[0]
            bb2yTL = bbx2[1]
            bb2xTR = bbx2[2]
            bb2yTR = bbx2[3]
            bb2xBR = bbx2[4]
            bb2yBR = bbx2[5]
            bb2xBL = bbx2[6]
            bb2yBL = bbx2[7]

            b2 = Polygon([(bb2xTL, bb2yTL), (bb2xTR, bb2yTR), (bb2xBR, bb2yBR), (bb2xBL, bb2yBL)])

            b = b1.intersection(b2)
            bcoords = b.exterior.coords.xy

            bi = []
            for x, y in zip(bcoords[0], bcoords[1]):
                bi.append([x, y])



            if len(bbox_indexes) > 2:
                for i in range(2,len(bbox_indexes)):
                    bbxNext = bbox_list[bbox_indexes[i]]
                    bbNextxTL = bbxNext[0]
                    bbNextyTL = bbxNext[1]
                    bbNextxTR = bbxNext[2]
                    bbNextyTR = bbxNext[3]
                    bbNextxBR = bbxNext[4]
                    bbNextyBR = bbxNext[5]
                    bbNextxBL = bbxNext[6]
                    bbNextyBL = bbxNext[7]
                    bbox_class = bbxNext[1]

                    bbNext = Polygon([(bbNextxTL, bbNextyTL), (bbNextxTR, bbNextyTR), (bbNextxBR, bbNextyBR), (bbNextxBL, bbNextyBL)])

                    try:
                        b = bbNext.intersection(b)

                    except:
                        print('hej')
                    bcoords = b.exterior.coords.xy

                    bi = []
                    for x, y in zip(bcoords[0], bcoords[1]):
                        bi.append([x, y])
                    center = b.centroid
                    bi.append(np.array(center))

                bi.append(cls)
                intersected_bboxes.append(bi)
                measurements.append((bi))

            else:
                center = b.centroid
                bi.append(np.array(center))
                bi.append(cls)
                intersected_bboxes.append(bi)
                measurements.append(bi)
        else:
            bbx1 = bbox_list[bbox_indexes[0]]

            bb1xTL = bbx1[0]
            bb1yTL = bbx1[1]
            bb1xTR = bbx1[2]
            bb1yTR = bbx1[3]
            bb1xBR = bbx1[4]
            bb1yBR = bbx1[5]
            bb1xBL = bbx1[6]
            bb1yBL = bbx1[7]

            center = [(min(bb1xTL, bb1xTR, bb1xBR, bb1xBL) + max(bb1xTL, bb1xTR, bb1xBR,bb1xBL)) / 2,
                      (min(bb1yTL, bb1yTR, bb1yBR, bb1yBL) + max(bb1yTL, bb1yTR, bb1yBR, bb1yBL)) / 2]
            bi = [[bb1xTL, bb1yTL], [bb1xTR, bb1yTR], [bb1xBR, bb1yBR], [bb1xBL, bb1yBL], [bb1xTL, bb1yTL], center, cls]
            intersected_bboxes.append(bi)
            measurements.append(bi)
    return intersected_bboxes, measurements


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



# Maps a list of bounding boxes from camera perspective into planar perspective.
def map_bboxes(bbox_list, mapObjects_list, classes_list):
    bbox_all_list = np.array([[1,2,3,4,5,6,7,8,9]], dtype=int)
    # Collect all bboxes
    for (bbox, mapObject, bbox_class) in zip(bbox_list, mapObjects_list, classes_list):
        if bbox != []:
            for i, box in enumerate(bbox):
                x1c, y1c, x2c, y2c = [int(i) for i in box]


                TL = np.array([[x1c, y1c]], dtype='float32')
                TR = np.array([[x2c, y1c]], dtype='float32')
                BR = np.array([[x2c, y2c]], dtype='float32')
                BL = np.array([[x1c, y2c]], dtype='float32')


                TL = np.array([TL])
                TR = np.array([TR])
                BR = np.array([BR])
                BL = np.array([BL])

                xTL, yTL = mapObject.getPoint(TL)
                xTR, yTR = mapObject.getPoint(TR)
                xBR, yBR = mapObject.getPoint(BR)
                xBL, yBL = mapObject.getPoint(BL)

                bbox_all_list = np.append(bbox_all_list, [[xTL, yTL, xTR, yTR, xBR, yBR, xBL, yBL, bbox_class]], axis=0)

    bbox_all_list = np.delete(bbox_all_list, (0), axis=0)
    return bbox_all_list