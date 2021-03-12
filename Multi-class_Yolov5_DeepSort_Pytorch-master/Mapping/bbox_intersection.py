import cv2
from Mapping.mapper import *
from shapely.geometry import Polygon


CLASSES = ['AGV', 'Human']
CAMERAS = ['','WN', 'MSW', 'NS', 'M', 'EN', 'ME', 'WN2', 'MW']

############################## POLYGON AREA ##############################
def areaOfPolygon(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))
def segments(p):
    return zip(p, p[1:] + [p[0]])

############################### DRAW BBOXES ###############################

def draw_multiple_boxes(bbox_list, mapping_objects, classes_list, cam_id_list, img):
    '''
    Input:  bbox_list: 2D list of bboxes
            mapping_objects: List of mapper class objects
            classes_list: 2D list with classes for each boundingbox
            cam_id_list: 2D list with camera id for each bounding box to know which camera it comes from
            img: Image to draw the boxes on

    Output: Image with boundingboxes
    '''

    classes_list = [item.tolist() for item in classes_list]
    for bbox, mapperObject, cls, cam_id in zip(bbox_list, mapping_objects, classes_list, cam_id_list):
        for (i, box), id in zip(enumerate(bbox), cam_id):
            x1, y1, x2, y2 = [int(i) for i in box]

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

            # Plot the boxes
            color = compute_color_for_labels(id)
            cv2.polylines(img, [pts], True, color, 2)
            label = str(CAMERAS[int(id)])
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (xTR, yTR), (xTR + t_size[0] + 3, yTR + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (xTR, yTR + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img

def draw_bboxes(bbox_list, img):
    '''
    Input:  bbox_list: 2D list of bboxes
            img: Image to draw on

    Output: img: Image with the boundingboxes on
    '''
    # Convert pixels to meter
    scale_x = 50 / 1788
    scale_y = 30 / 1069

    for bbox in bbox_list:
        center = bbox[-2]
        centerX = round(scale_x * center[0], 3)
        centerY = round((1069 - center[1]) * scale_y, 3)

        pts = np.array([])
        for i in range(len(bbox) - 1):
            pts = np.append(pts, np.round(bbox[i]))
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [np.int32(pts)], True, (255, 0, 0), 2)
        cv2.circle(img, (np.int32(center[0]), np.int32(center[1])),2, (255,0,0),2)
        label = str('x = ' + str(centerX) + ' y = ' + str(centerY))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

        x = round(bbox[0][0])
        y = round(bbox[0][1])

        cv2.putText(img, label, (x, y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

############################### INTERSECTION ###############################

def compute_iou_matrix(bbox_list, mapObjects_list, camera_id_list):
    '''
    Input:  bbox_list: List of N bboxes
            mapObjects_list: List with mapper class objects
            camera_id_list: List with camera id for each bbox

    Output: iou_matrix: NxN matrix with IoU-value
    '''
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

        areaBbx1 = areaOfPolygon([[bb1xTL, bb1yTL], [bb1xTR, bb1yTR], [bb1xBR, bb1yBR], [bb1xBL, bb1yBL]])
        # determine the coordinates of the intersection rectangle

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

            if p1.intersects(p2):
                intersection = p1.intersection(p2)

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
                iou_matrix[i,j] = iou

    return iou_matrix

def find_intersections(bbox_list, mapObjects_list, classes_list, camera_id_list):
    '''
    Input:  bbox_list: List of N bboxes
            mapObjects_list: List with mapper class objects
            classes_list: List of classes for each bbox
            camera_id_list: List with camera id for each bbox

    Output: iou_matrix: NxN matrix with IoU-value
    '''
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

            intersecting_bboxes.append(intersected_bbox)
            cls_list.append(class_bbox)
    return intersecting_bboxes, cls_list

def compute_multiple_intersection_bboxes(intersecting_bboxes, bbox_listed, class_list):
    '''
    Input:
            bbox_list: List of N bboxes
            mapObjects_list: List with mapper class objects
            camera_id_list: List with camera id for each bbox

    Output: iou_matrix: NxN matrix with IoU-value
    '''
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

########################## MAPPING #########################

# Maps a list of bounding boxes from camera perspective into planar perspective.
def map_bboxes(bbox_list, mapObjects_list, classes_list):
    '''
    Input:
            bbox_list: List of N bboxes
            mapObjects_list: List with mapper class objects
            classes_list: List of classes for each bbox

    Output: bbox_all_list: List of mapped bboxes
    '''
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

def bbox_to_coords(bbox_list, mapObjects_list, camera_id_list):
    '''
    Input:
            bbox_list: List of N bboxes
            mapObjects_list: List with mapper class objects
            camera_id_list: List with camera id for each bbox

    Output: bbox_all_list: List of mapped bboxes
    '''
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

######################### HELPER FUNCTIONS #################

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)