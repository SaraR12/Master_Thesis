import threading, queue

from Mapping.bbox_intersection import *
from Mapping.opticalflow import *
from homographies import *
from KalmanTracker import *
import trackNoDeepSort
import matplotlib.pyplot as plt
from CollisionAvoidance.safety_zone import getSafetyZone


############################################## MAIN FILE ######################################################
# Main file, run this file

qWNList = []
qMSWList = []
qNSList = []
qMList = []
qENList = []
qMEList = []
qWN2List = []


# Create functions for running each camera. One list q for each camera
def trackerCamWN(path):
    camera = 'WN'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qWNList.append([i])

def trackerCamMSW(path):
    camera = 'MSW'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qMSWList.append([i])

def trackerCamNS(path):
    camera = 'NS'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qNSList.append([i])

def trackerCamM(path):
    camera = 'M'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qMList.append([i])

def trackerCamEN(path):
    camera = 'EN'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qENList.append([i])

def trackerCamME(path):
    camera = 'ME'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qMEList.append([i])

def trackerCamWN2(path):
    camera = 'MW'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qWN2List.append([i])

# Crate consumer that is the main execution function
def consumer():
    # Get overview image
    PLANE = cv2.imread('Mapping/plane.png')
    CAP = cv2.VideoCapture('videos/VideoOrto.mkv')
    ret, VIDEOFRAME = CAP.read()
    VIDEOFRAME = cv2.resize(VIDEOFRAME, (1788, 1069))

    # Get all the homographies to the different cameras
    pts_src, pts_dst = getKeypoints('WN')
    mapObjWN = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("MSW")
    mapObjMSW = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("NS")
    mapObjNS = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("M")
    mapObjM = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("EN")
    mapObjEN = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints('ME')
    mapObjME = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints('MW')
    mapObjWN2 = Mapper(PLANE, pts_src, pts_dst)

    mapping_objects = [mapObjWN, mapObjMSW, mapObjNS, mapObjM, mapObjEN, mapObjME, mapObjWN2]

    output1 = []
    outputFiltered1 = []
    i = 0
    frame = 1
    plt.show()

    while i < 300:

        lenWN = len(qWNList)
        lenMSW = len(qMSWList)
        lenNS = len(qNSList)
        lenM = len(qMList)
        lenEN = len(qENList)
        lenME = len(qMEList)
        lenWN2 = len(qWN2List)

        # Want to secure that all produces has produced something so we don't run on empty lists
        i = min(lenWN, lenMSW, lenNS, lenM, lenEN, lenME, lenWN2)
        if i > 0:
            classesWN, classesMSW, classesNS, classesM, classesEN, classesME, classesWN2 = np.array([]), np.array([]), \
                                                                                            np.array([]),np.array([]), np.array([]), np.array([]), np.array([])

            if all([qWNList, qMSWList, qNSList, qMList, qENList, qMEList, qWN2List]): # True if all not 0
                # Create lists with bbox coords and classes for the detected objects
                qWNValue = qWNList.pop(0)
                if qWNValue[0] is not None:
                    bbox_xyxyWN = qWNValue[0][:][:,:4]
                    classesWN = qWNValue[0][:][:,5]
                else:
                    bbox_xyxyWN = None

                qMSWValue = qMSWList.pop(0)
                if qMSWValue[0] is not None:
                    bbox_xyxyMSW = qMSWValue[0][:][:,:4]
                    classesMSW = qMSWValue[0][:][:,5]

                else:
                    bbox_xyxyMSW = None

                qNSValue = qNSList.pop(0)
                if qNSValue[0] is not None:
                    bbox_xyxyNS = qNSValue[0][:][:,:4]
                    classesNS = qNSValue[0][:][:,5]
                else:
                    bbox_xyxyNS = None

                qMValue = qMList.pop(0)
                if qMValue[0] is not None:
                    bbox_xyxyM = qMValue[0][:][:,:4]
                    classesM = qMValue[0][:][:,5]
                else:
                    bbox_xyxyM = None

                qENValue = qENList.pop(0)
                if qENValue[0] is not None:
                    bbox_xyxyEN = qENValue[0][:][:,:4]
                    classesEN = qENValue[0][:][:,5]
                else:
                    bbox_xyxyEN = None

                qMEValue = qMEList.pop(0)
                if qMEValue[0] is not None:
                    bbox_xyxyME = qMEValue[0][:][:,:4]
                    classesME = qMEValue[0][:][:,5]
                else:
                    bbox_xyxyME = None

                qWN2Value = qWN2List.pop(0)
                if qWN2Value[0] is not None:
                    bbox_xyxyWN2 = qWN2Value[0][:][:, :4]
                    classesWN2 = qWN2Value[0][:][:, 5]
                else:
                    bbox_xyxyWN2 = None


                bbox_list = [bbox_xyxyWN if bbox_xyxyWN is not None else [],
                             bbox_xyxyMSW if bbox_xyxyMSW is not None else [],
                             bbox_xyxyNS if bbox_xyxyNS is not None else [],
                             bbox_xyxyM if bbox_xyxyM is not None else [],
                             bbox_xyxyEN if bbox_xyxyEN is not None else [],
                             bbox_xyxyME if bbox_xyxyME is not None else [],
                             bbox_xyxyWN2 if bbox_xyxyWN2 is not None else []]


                cam_id_list = [[np.ones(len(bbox_xyxyWN if bbox_xyxyWN is not None else []))[:].tolist()] +
                               [(np.ones(len(bbox_xyxyMSW if bbox_xyxyMSW is not None else []))*2)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyNS if bbox_xyxyNS is not None else []))*3)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyM if bbox_xyxyM is not None else []))*4)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyEN if bbox_xyxyEN is not None else []))*5)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyME if bbox_xyxyME is not None else []))*6)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyWN2 if bbox_xyxyWN2 is not None else []))*7)[:].tolist()]][0]


                classes_list = [classesWN.tolist() + classesMSW.tolist() + classesNS.tolist() +
                                classesM.tolist() + classesEN.tolist() + classesME.tolist() + classesWN2.tolist()][0]

                # Somehow when we trained the network class 2 and 0 was AGV and 3 and 1 Human
                for i, cls in enumerate(classes_list):
                    if cls == 2:
                        classes_list[i] = 0
                    elif cls == 3:
                        classes_list[i] = 1

                # Want to plot all boundinboxes on one image and only plot kalman on the other so copy the image we want to plot on
                VIDEOFRAME2 = np.copy(VIDEOFRAME)

                # Draw all projected boxes
                img = draw_multiple_boxes(bbox_list, mapping_objects,
                                          [classesWN, classesMSW, classesNS, classesM, classesEN, classesME, classesWN2], cam_id_list, VIDEOFRAME2)

                # Find what boundingboxes that intersect
                intersecting_bboxes, intersecting_classes_list = find_intersections(bbox_list, mapping_objects, classes_list, cam_id_list)

                # Map the projected bboxes, intersect and plot them
                bbox_all_list = map_bboxes(bbox_list, mapping_objects, classes_list)
                intersected_bboxes, measurements = compute_multiple_intersection_bboxes(intersecting_bboxes, bbox_all_list, classes_list)
                #img2 = draw_bboxes(intersected_bboxes, VIDEOFRAME)
                img2 = VIDEOFRAME
                #################################### KALMAN FILTERING ######################################################

                if frame == 4:
                    output1.append(measurements[0][-2])


                    bbox_xyah, classlist_bbox = preprocessMeasurements(measurements)
                    #filter_list, mean_list, covariance_list = InitKalmanTracker(bbox_xyah)
                    filter_listUKF, x_list = InitUKFTracker(bbox_xyah)
                    opticalflow_list = []

                    for (id, bbox), cls in zip(enumerate(bbox_xyah), classlist_bbox):
                        opticalflow_list.append(OpticalFlow(frame, id, bbox, cls))

                elif frame > 4:
                    bbox_xyah, classlist_bbox = preprocessMeasurements(measurements)
                    heading_list = []
                    for (id,bbox), cls, opflow in zip(enumerate(bbox_xyah), classlist_bbox, opticalflow_list):
                        state = opflow(frame, id, bbox, cls)
                        heading_list.append(state[0])
                        print('state', state)

                    #filter_list, mean_list, covariance_list = predictKalmanTracker(filter_list, mean_list, covariance_list)
                    filter_listUKF, x_list = predictUKFTracker(filter_listUKF, x_list)

                    bbox_xyah, classlist_bbox = association(filter_listUKF, x_list, bbox_xyah, classlist_bbox)

                    #output1.append(calculateCenterPoint(bbox_xyah[0]))

                    #filter_list, mean_list, covariance_list = updateKalmanTracker(filter_list, mean_list, covariance_list, bbox_xyah)
                    filter_listUKF, x_list = updateUKFTracker(filter_listUKF, x_list, bbox_xyah)
                    #outputFiltered1.append(calculateCenterPoint(mean_list[0]))

                    points_list = getSafetyZone(x_list,heading_list, classlist_bbox)
                    img2 = draw_bboxes(points_list, img2)

                    # Draw filter outputs
                    #img2 = drawFilterOutput(x_list, img2)

                ######################################## SHOW RESULTS ######################################################
                print('Frame: ', frame)
                cv2.imshow('overview', img)
                cv2.imshow('Intersected', img2)

                # To step through frames
                if cv2.waitKey(0) == 33:
                    continue
                frame += 1
                if frame == 300:
                    for xy in output1:
                        plt.plot(xy[0], xy[1 ])
                ret, VIDEOFRAME = CAP.read()
                VIDEOFRAME = cv2.resize(VIDEOFRAME, (1788, 1069))

if __name__ == '__main__':
    qWN = queue.Queue()
    qMSW = queue.Queue()
    qNS = queue.Queue()
    qM = queue.Queue()
    qEN = queue.Queue()
    qME = queue.Queue()
    qWN2 = queue.Queue()

    # Producers
    threadCamWN = threading.Thread(target=trackerCamWN, args=('videos/VideoWN.mkv',), daemon=True).start() #1
    threadCamMSW = threading.Thread(target=trackerCamMSW, args=('videos/VideoMSW.mkv',), daemon=True).start() # 2
    threadCamNS = threading.Thread(target=trackerCamNS, args=('videos/VideoNS.mkv',), daemon=True).start() #3
    threadCamM = threading.Thread(target=trackerCamM, args=('videos/VideoM.mkv',), daemon=True).start() #4
    threadCamEN = threading.Thread(target=trackerCamEN, args=('videos/VideoEN.mkv',), daemon=True).start() #5
    threadCamME = threading.Thread(target=trackerCamME, args=('videos/VideoME.mkv',), daemon=True).start() # 6
    threadCamWN2 = threading.Thread(target=trackerCamWN2, args=('videos/VideoMW.mkv',), daemon=True).start()  # 7

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()
