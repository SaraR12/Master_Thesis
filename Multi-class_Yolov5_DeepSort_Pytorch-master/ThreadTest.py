import threading, queue

from Mapping.bbox_intersection import *
from Mapping.opticalflow import *
from homographies import *
from KalmanTracker import *
from CollisionAvoidance.heatmap import HeatMap
import trackNoDeepSort
import matplotlib.pyplot as plt
from CollisionAvoidance.safety_zone import getSafetyZone
import time
from Mapping.positioning_evaluation import filtered_positions


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
    camera = 'BL'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qWNList.append([i])

def trackerCamMSW(path):
    camera = 'ML2'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qMSWList.append([i])

def trackerCamNS(path):
    camera = 'MM'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qNSList.append([i])

def trackerCamM(path):
    camera = 'MR'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qMList.append([i])

def trackerCamEN(path):
    camera = 'TL2'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qENList.append([i])

def trackerCamME(path):
    camera = 'TR'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qMEList.append([i])

def trackerCamWN2(path):
    camera = 'CameraShelf'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qWN2List.append([i])

# Crate consumer that is the main execution function
def consumer():
    # Get overview image
    PLANE = cv2.imread('Mapping/plane1.png')
    H, W, _ = PLANE.shape
    CAP = cv2.VideoCapture('videos/VideoOrtoLargeCanvas.mkv')
    VIDEOFRAME = PLANE
    ret, VIDEOFRAME = CAP.read()
    VIDEOFRAME = cv2.resize(VIDEOFRAME, (W, H))

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

    pts_src, pts_dst = getKeypoints('WN2')
    mapObjWN2 = Mapper(PLANE, pts_src, pts_dst)

    mapping_objects = [mapObjWN, mapObjMSW, mapObjNS, mapObjM, mapObjEN, mapObjME, mapObjWN2]  # mapObjWN2

    output1 = []
    outputFiltered1 = []
    i = 0
    frame = 1
    plt.show()


    prev_time = time.time()
    heatmap_obj = HeatMap(VIDEOFRAME, timesteps=20)

    while i < 300:

        lenWN = len(qWNList)
        lenMSW = len(qMSWList)
        lenNS = len(qNSList)
        lenM = len(qMList)
        lenEN = len(qENList)
        lenME = len(qMEList)
        lenWN2 = len(qWN2List)

        # Want to secure that all produces has produced something so we don't run on empty lists
        i = min(lenWN, lenMSW, lenNS, lenM, lenEN, lenME, lenWN2)  # lenWN2
        print(i)
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
                               [(np.ones(len(bbox_xyxyEN if bbox_xyxyEN is not None else [])) * 5)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyME if bbox_xyxyME is not None else []))*6)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyWN2 if bbox_xyxyWN2 is not None else []))*7)[:].tolist()]][0]


                classes_list = [classesWN.tolist() + classesMSW.tolist() + classesNS.tolist() +
                                classesM.tolist() + classesEN.tolist() + classesME.tolist() + classesWN2.tolist()][0]

                # Want to plot all boundinboxes on one image and only plot kalman on the other so copy the image we want to plot on
                VIDEOFRAME2 = np.copy(VIDEOFRAME)

                # Draw all projected boxes
                img = draw_multiple_boxes(bbox_list, mapping_objects,
                                          [classesWN, classesMSW, classesNS, classesM, classesEN, classesME], cam_id_list, VIDEOFRAME2)



                # Map the projected bboxes, intersect and plot them
                bbox_all_list, classes_list = map_bboxes(bbox_list, mapping_objects, classes_list)

                # Find what boundingboxes that intersect
                intersecting_bboxes, intersecting_classes_list = find_intersections(bbox_all_list, mapping_objects, classes_list, cam_id_list)

                intersected_bboxes, measurements = compute_multiple_intersection_bboxes(intersecting_bboxes, bbox_all_list, classes_list)
                #img2 = draw_bboxesTEMP(intersected_bboxes, VIDEOFRAME)
                #cv2.imshow('Intersected', img2)
                img2 = VIDEOFRAME
                #################################### KALMAN FILTERING ######################################################

                if frame == 1:
                    output1.append(measurements[0][-2])


                    bbox_xyah, classlist_bbox = preprocessMeasurements(measurements)
                    #filter_list, mean_list, covariance_list = InitKalmanTracker(bbox_xyah)
                    filter_listUKF, x_list = InitUKFTracker(bbox_xyah, classlist_bbox)
                    opticalflow_list = []

                    for (id, point), cls in zip(enumerate(x_list), classlist_bbox):
                        opticalflow_list.append(OpticalFlow(frame, id, point, cls))
                    """for (id, point), cls, filter in zip(enumerate(x_list), classlist_bbox, filter_listUKF):
                        filter.InitOpticalFlow(frame, id, point, cls)"""

                elif frame > 1:
                    bbox_xyah, classlist_bbox = preprocessMeasurements(measurements)
                    heading_list = []

                    filter_listUKF, x_list, classlist_bbox_test = predictUKFTracker(filter_listUKF, x_list)

                    bbox_xyah, unassociated_bbox_xyah, unassociated_class_list = association(filter_listUKF, x_list, bbox_xyah, classlist_bbox)
                    if unassociated_bbox_xyah != []:
                        filter_listUKFnew, x_listnew = InitUKFTracker(unassociated_bbox_xyah, unassociated_class_list, filterlist=filter_listUKF)


                        for (id, point), cls in zip(enumerate(x_listnew), unassociated_class_list):
                            opticalflow_list.append(OpticalFlow(frame, id, point, cls))
                        filter_listUKF.append(filter_listUKFnew[0])
                        x_list.append(x_listnew[0])
                        #bbox_xyah.append(unassociated_bbox_xyah[0])

                    filter_listUKF, x_list, opticalflow_list = updateUKFTracker(filter_listUKF, x_list, bbox_xyah, opticalflow_list)
                    #heatmap = heatmap_obj.update(x_list, classlist_bbox_test)
                    filtered_positions(x_list, frame)

                    for (id, point), cls, opflow in zip(enumerate(x_list), classlist_bbox_test, opticalflow_list):
                        state = opflow(frame, id, point, cls)
                        heading_list.append(state[0])
                        cv2.circle(img2, (int(point[0]), int(point[1])), 1, (0,0,255), 2)

                    ######################################## SHOW RESULTS ######################################################
                    #cv2.imshow('heatmap', heatmap)
                    #heatmapshow = None
                    #heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                    #                            dtype=cv2.CV_8U)
                    #heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_TURBO)
                    #cv2.imshow("Heatmap", heatmapshow)

                    points_list = getSafetyZone(x_list, heading_list, classlist_bbox_test)
                    img2 = draw_bboxes(points_list, img2, filter_listUKF)


                new_time = time.time()
                fps = frame/(new_time - prev_time)
                fps = round(fps,2)
                print('Frame: ', frame)
                cv2.putText(img2,str(fps),(7,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0),3,cv2.LINE_AA)
                cv2.imshow('overview', img)
                cv2.imshow('Intersected', img2)


                # To step through frames
                """if cv2.waitKey(0) == 33:
                    continue"""
                cv2.waitKey(1)
                frame += 1
                """if frame == 293:
                    print('xdrift')"""

                ret, VIDEOFRAME = CAP.read()
                VIDEOFRAME = cv2.resize(VIDEOFRAME, (W, H))

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
    threadCamWN2 = threading.Thread(target=trackerCamWN2, args=('videos/VideoWN2.mkv',), daemon=True).start()  # 7

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()