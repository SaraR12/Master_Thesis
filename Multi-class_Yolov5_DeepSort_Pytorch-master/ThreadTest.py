import threading, queue

from Mapping.bbox_intersection import *
from homographies import *
from KalmanTracker import *
import trackNoDeepSort
#from deep_sort.deep_sort.sort.tracker import *

qWNList = []
qMSWList = []
qNSList = []
qMList = []
qENList = []
qMEList = []
qWN2List = []
#qMWList = []

def tlwh_to_xyxy(bbox_tlwh):
    """
    TODO:
        Convert bbox from xtl_ytl_w_h to xc_yc_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    x, y, w, h = bbox_tlwh
    x1 = int(round(x))
    x2 = x1 + int(round(w))
    y1 = int(round(y))
    y2 = y1 + int(round(h))
    return x1, y1, x2, y2


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
    camera = 'MW'
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
    camera = 'WN2'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qWN2List.append([i])

'''def trackerCamMW(path):
    camera = 'MW'
    out = trackNoDeepSort.run(path, camera)
    for i in out:
        qMWList.append([i])'''

def consumer():
    PLANE = cv2.imread('Mapping/plane.png')

    CAP = cv2.VideoCapture('videos/VideoOrto.mkv')
    ret, VIDEOFRAME = CAP.read()
    VIDEOFRAME = cv2.resize(VIDEOFRAME, (1788, 1069))

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

    '''pts_src, pts_dst = getKeypoints('MW')
    mapObjMW = Mapper(PLANE, pts_src, pts_dst)'''

    mapping_objects = [mapObjWN, mapObjMSW, mapObjNS, mapObjM, mapObjEN, mapObjME, mapObjWN2] #, mapObjMW]

    calculated_frames = []
    output1 = []
    outputFiltered1 = []
    i = 0
    frame = 1
    plt.show()

    while i < 300:
        #
        lenWN = len(qWNList)
        lenMSW = len(qMSWList)
        lenNS = len(qNSList)
        lenM = len(qMList)
        lenEN = len(qENList)
        lenME = len(qMEList)
        lenWN2 = len(qWN2List)
        #lenMW = len(qMWList)
        i = min(lenWN, lenMSW, lenNS, lenM, lenEN, lenME, lenWN2)  # , lenMW
        if i > 0:
            classesWN, classesMSW, classesNS, classesM, classesEN, classesME, classesWN2 = np.array([]), np.array([]), \
                                                                                            np.array([]),np.array([]), np.array([]), np.array([]), np.array([])
            confWN, confMSW, confNS, confM, confEN, confME, confWN2 = np.array([]), np.array([]), \
                                                                                            np.array([]),np.array([]), np.array([]), np.array([]), np.array([])
            #qWNList,
            if all([qMSWList,qNSList,qMList,qENList, qMEList, qWN2List]):
                qWNValue = qWNList.pop(0)
                if qWNValue[0] is not None:
                    bbox_xyxyWN = qWNValue[0][:][:,:4]
                    identitiesWN = np.ones(len(qWNValue[0][:][:,4]))
                    classesWN = qWNValue[0][:][:,5]
                    confWN = qWNValue[0][:][:,6]
                else:
                    bbox_xyxyWN = None
                    identitiesWN = None

                qMSWValue = qMSWList.pop(0)
                if qMSWValue[0] is not None:
                    bbox_xyxyMSW = qMSWValue[0][:][:,:4]
                    identitiesMSW = np.ones(len(qMSWValue[0][:][:,4])) * 2
                    classesMSW = qMSWValue[0][:][:,5]
                    confMSW = qMSWValue[0][:][:,6]

                else:
                    bbox_xyxyMSW = None
                    identitiesMSW = None

                qNSValue = qNSList.pop(0)
                if qNSValue[0] is not None:
                    bbox_xyxyNS = qNSValue[0][:][:,:4]
                    identitiesNS = np.ones(len(qNSValue[0][:][:,4])) * 3
                    classesNS = qNSValue[0][:][:,5]
                    confNS = qNSValue[0][:][:,6]
                else:
                    bbox_xyxyNS = None
                    identitiesNS = None

                qMValue = qMList.pop(0)
                if qMValue[0] is not None:
                    bbox_xyxyM = qMValue[0][:][:,:4]
                    identitiesM = np.ones(len(qMValue[0][:][:,4])) * 4
                    classesM = qMValue[0][:][:,5]
                    confM = qMValue[0][:][:,6]
                else:
                    bbox_xyxyM = None
                    identitiesM = None

                qENValue = qENList.pop(0)
                if qENValue[0] is not None:
                    bbox_xyxyEN = qENValue[0][:][:,:4]
                    identitiesEN = np.ones(len(qENValue[0][:][:,4])) * 5
                    classesEN = qENValue[0][:][:,5]
                    confEN = qENValue[0][:][:,6]
                else:
                    bbox_xyxyEN = None
                    identitiesEN = None

                qMEValue = qMEList.pop(0)
                if qMEValue[0] is not None:
                    bbox_xyxyME = qMEValue[0][:][:,:4]
                    identitiesME = np.ones(len(qMEValue[0][:][:,4])) * 6
                    classesME = qMEValue[0][:][:,5]
                    confME = qMEValue[0][:][:,6]
                else:
                    bbox_xyxyME = None
                    identitiesME = None

                qWN2Value = qWN2List.pop(0)
                if qWN2Value[0] is not None:
                    bbox_xyxyWN2 = qWN2Value[0][:][:, :4]
                    identitiesWN2 = np.ones(len(qWN2Value[0][:][:, 4])) * 7
                    classesWN2 = qWN2Value[0][:][:, 5]
                    confWN2 = qWN2Value[0][:][:, 6]
                else:
                    bbox_xyxyWN2 = None
                    identitiesWN2 = None

                '''qMWValue = qMWList.pop(0)
                if qMWValue[0] is not None:
                    bbox_xyxyMW = qMWValue[0][:][:, :4]
                    identitiesMW = np.ones(len(qMWValue[0][:][:, 4])) * 8
                    classesMW = qMWValue[0][:][:, 5]
                    confMW = qMWValue[0][:][:, 6]

                else:
                    bbox_xyxyMW = None
                    identitiesMW = None'''


                bbox_list = [bbox_xyxyWN if bbox_xyxyWN is not None else [],
                             bbox_xyxyMSW if bbox_xyxyMSW is not None else [],
                             bbox_xyxyNS if bbox_xyxyNS is not None else [],
                             bbox_xyxyM if bbox_xyxyM is not None else [],
                             bbox_xyxyEN if bbox_xyxyEN is not None else [],
                             bbox_xyxyME if bbox_xyxyME is not None else [],
                             bbox_xyxyWN2 if bbox_xyxyWN2 is not None else []]
                             #bbox_xyxyMW if bbox_xyxyMW is not None else []]

                cam_id_list = [[np.ones(len(bbox_xyxyWN if bbox_xyxyWN is not None else []))[:].tolist()] +
                               [(np.ones(len(bbox_xyxyMSW if bbox_xyxyMSW is not None else []))*2)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyNS if bbox_xyxyNS is not None else []))*3)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyM if bbox_xyxyM is not None else []))*4)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyEN if bbox_xyxyEN is not None else []))*5)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyME if bbox_xyxyME is not None else []))*6)[:].tolist()] +
                               [(np.ones(len(bbox_xyxyWN2 if bbox_xyxyWN2 is not None else []))*7)[:].tolist()]][0]
                               #[(np.ones(len(bbox_xyxyMW if bbox_xyxyMW is not None else []))*8)[:].tolist()]][0]

                classes_list = [classesWN.tolist() + classesMSW.tolist() + classesNS.tolist() +
                                classesM.tolist() + classesEN.tolist() + classesME.tolist() + classesWN2.tolist()][0] #+ classesMW.tolist()]

                for i, cls in enumerate(classes_list):
                    if cls == 2:
                        classes_list[i] = 0
                    elif cls == 3:
                        classes_list[i] = 1

                identities_list = [identitiesWN.tolist() if identitiesWN is not None else None,
                                   identitiesMSW.tolist() if identitiesMSW is not None else None,
                                   identitiesNS.tolist() if identitiesNS is not None else None,
                                   identitiesM.tolist() if identitiesM is not None else None,
                                   identitiesEN.tolist() if identitiesEN is not None else None,
                                   identitiesME.tolist() if identitiesME is not None else None,
                                   identitiesWN2.tolist() if identitiesWN2 is not None else None]
                                   #identitiesMW.tolist() if identitiesMW is not None else None]

                """conf_list = [confWN.tolist() + confMSW.tolist() + confNS.tolist() + confM.tolist() + confEN.tolist() + 
                             confME.tolist() + confWN2.tolist() + confMW.tolist()][0]"""
                VIDEOFRAME2 = np.copy(VIDEOFRAME)

                img = draw_multiple_boxes(bbox_list, mapping_objects, identities_list,
                                          [classesWN, classesMSW, classesNS, classesM, classesEN, classesME, classesWN2], cam_id_list, VIDEOFRAME2)

                intersecting_bboxes, intersecting_classes_list = find_intersections(bbox_list, mapping_objects, classes_list, cam_id_list)

                bbox_all_list = map_bboxes(bbox_list, mapping_objects, classes_list)

                intersected_bboxes, measurements = compute_multiple_intersection_bboxes(intersecting_bboxes, bbox_all_list, classes_list)

                img2 = draw_bboxes(intersected_bboxes, VIDEOFRAME)

                #t3 = time_synchronized()
                KF = True

                if KF:

                    #################################### KALMAN FILTERING ######################################################

                    if frame == 4:
                        output1.append(measurements[0][-2])
                        bbox_xyah = preprocessMeasurements(measurements)
                        filter_list, mean_list, covariance_list = InitKalmanTracker(bbox_xyah)
                    elif frame > 4:

                        bbox_xyah = preprocessMeasurements(measurements)

                        filter_list, mean_list, covariance_list = predictKalmanTracker(filter_list, mean_list, covariance_list)

                        bbox_xyah = association(filter_list, mean_list, covariance_list, bbox_xyah)

                        output1.append(calculateCenterPoint(bbox_xyah[0]))

                        filter_list, mean_list, covariance_list = updateKalmanTracker(filter_list, mean_list, covariance_list, bbox_xyah)

                        outputFiltered1.append(calculateCenterPoint(mean_list[0]))

                        # Draw filter outputs
                        drawFilterOutput(mean_list, img2)

                    ######################################## SHOW RESULTS ######################################################
                print('Frame: ', frame)
                cv2.imshow('overview', img)
                cv2.imshow('Intersected', img2)


                calculated_frames.append(i)
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
    #qMW = queue.Queue()

    # Producers
    threadCamWN = threading.Thread(target=trackerCamWN, args=('videos/VideoWN.mkv',), daemon=True).start() #1
    threadCamMSW = threading.Thread(target=trackerCamMSW, args=('videos/VideoMSW.mkv',), daemon=True).start() # 2
    threadCamNS = threading.Thread(target=trackerCamNS, args=('videos/VideoNS.mkv',), daemon=True).start() #3
    threadCamM = threading.Thread(target=trackerCamM, args=('videos/VideoM.mkv',), daemon=True).start() #4
    threadCamEN = threading.Thread(target=trackerCamEN, args=('videos/VideoEN.mkv',), daemon=True).start() #5
    threadCamME = threading.Thread(target=trackerCamME, args=('videos/VideoME.mkv',), daemon=True).start() # 6
    threadCamWN2 = threading.Thread(target=trackerCamWN2, args=('videos/VideoMW.mkv',), daemon=True).start()  # 7
    #threadCamMW = threading.Thread(target=trackerCamMW, args=('videos/VideoMW.mkv',), daemon=True).start()  # 8

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()