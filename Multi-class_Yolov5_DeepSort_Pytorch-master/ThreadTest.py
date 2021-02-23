import threading, queue

import track
from Mapping.mapper import *
from Mapping.bbox_intersection import *
from homographies import *
from KalmanTracker import *

qWNList = []
qMSWList = []
qNSList = []
qMWList = []
qENList = []
qWN2List = []


def trackerCamWN(path):
    camera = 'WN'
    out = track.run(path, camera)
    for i in out:
        qWNList.append([i])

def trackerCamMSW(path):
    camera = 'MSW'
    out = track.run(path, camera)
    for i in out:
        qMSWList.append([i])

def trackerCamNS(path):
    camera = 'NS'
    out = track.run(path, camera)
    for i in out:
        qNSList.append([i])

def trackerCamMW(path):
    camera = 'MW'
    out = track.run(path, camera)
    for i in out:
        qMWList.append([i])

def trackerCamEN(path):
    camera = 'EN'
    out = track.run(path, camera)
    for i in out:
        qENList.append([i])

def trackerCamWN2(path):
    camera = 'WN2'
    out = track.run(path, camera)
    for i in out:
        qWN2List.append([i])

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

    pts_src, pts_dst = getKeypoints("WN2")
    mapObjWN2 = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("MW")
    mapObjMW = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("EN")
    mapObjEN = Mapper(PLANE, pts_src, pts_dst)

    mapping_objects = [mapObjWN, mapObjMSW, mapObjNS, mapObjWN2, mapObjMW, mapObjEN]

    calculated_frames = []
    i = 0
    frame = 1
    while i < 300:
        i = min(len(qWNList), len(qMSWList), len(qNSList), len(qWN2List), len(qMWList), len(qENList))

        classesWN, classesMSW, classesNS, classesWN2, classesMW, classesEN = np.array([]), np.array([]), np.array([]),\
                                                                     np.array([]), np.array([]), np.array([])
        if all([qWNList,qMSWList,qNSList,qWN2List,qMWList,qENList]):
            qWNValue = qWNList.pop(0)
            if qWNValue[0] is not None:
                bbox_xyxyWN = qWNValue[0][:][:,:4]
                identitiesWN = np.ones(len(qWNValue[0][:][:,4]))
                classesWN = qWNValue[0][:][:,5]
            else:
                bbox_xyxyWN = None
                identitiesWN = None

            qMSWValue = qMSWList.pop(0)
            if qMSWValue[0] is not None:
                bbox_xyxyMSW = qMSWValue[0][:][:,:4]
                identitiesMSW = np.ones(len(qMSWValue[0][:][:,4])) * 2
                classesMSW = qMSWValue[0][:][:,5]
            else:
                bbox_xyxyMSW = None
                identitiesMSW = None

            qNSValue = qNSList.pop(0)
            if qNSValue[0] is not None:
                bbox_xyxyNS = qNSValue[0][:][:,:4]
                identitiesNS = np.ones(len(qNSValue[0][:][:,4])) * 3
                classesNS = qNSValue[0][:][:,5]
            else:
                bbox_xyxyNS = None
                identitiesNS = None

            qWN2Value = qWN2List.pop(0)
            if qWN2Value[0] is not None:
                bbox_xyxyWN2 = qWN2Value[0][:][:,:4]
                identitiesWN2 = np.ones(len(qWN2Value[0][:][:,4])) * 4
                classesWN2 = qWN2Value[0][:][:,5]
            else:
                bbox_xyxyWN2 = None
                identitiesWN2 = None

            qMWValue = qMWList.pop(0)
            if qMWValue[0] is not None:
                bbox_xyxyMW = qMWValue[0][:][:,:4]
                identitiesMW = np.ones(len(qMWValue[0][:][:,4])) * 5
                classesMW = qMWValue[0][:][:,5]
            else:
                bbox_xyxyMW = None
                identitiesMW = None

            qENValue = qENList.pop(0)
            if qENValue[0] is not None:
                bbox_xyxyEN = qENValue[0][:][:,:4]
                identitiesEN = np.ones(len(qENValue[0][:][:,4])) * 6
                classesEN = qENValue[0][:][:,5]
            else:
                bbox_xyxyEN = None
                identitiesEN = None


            bbox_list = [bbox_xyxyWN if bbox_xyxyWN is not None else [],
                         bbox_xyxyMSW if bbox_xyxyMSW is not None else [],
                         bbox_xyxyNS if bbox_xyxyNS is not None else [],
                         bbox_xyxyWN2 if bbox_xyxyWN2 is not None else [],
                         bbox_xyxyMW if bbox_xyxyMW is not None else [],
                         bbox_xyxyEN if bbox_xyxyEN is not None else []]

            cam_id_list = [[np.ones(len(bbox_xyxyWN if bbox_xyxyWN is not None else []))[:].tolist()] +
                           [(np.ones(len(bbox_xyxyMSW if bbox_xyxyMSW is not None else []))*2)[:].tolist()] +
                           [(np.ones(len(bbox_xyxyNS if bbox_xyxyNS is not None else []))*3)[:].tolist()] +
                           [(np.ones(len(bbox_xyxyWN2 if bbox_xyxyWN2 is not None else []))*4)[:].tolist()] +
                           [(np.ones(len(bbox_xyxyMW if bbox_xyxyMW is not None else []))*5)[:].tolist()] +
                           [(np.ones(len(bbox_xyxyEN if bbox_xyxyEN is not None else []))*6)[:].tolist()]
                           ][0]

            classes_list = [classesWN.tolist() + classesMSW.tolist() + classesNS.tolist() + classesWN2.tolist() +
                            classesMW.tolist() + classesEN.tolist()][0]

            identities_list = [identitiesWN.tolist() if identitiesWN is not None else None,
                               identitiesMSW.tolist() if identitiesMSW is not None else None,
                               identitiesNS.tolist() if identitiesNS is not None else None,
                               identitiesWN2.tolist() if identitiesWN2 is not None else None,
                               identitiesMW.tolist() if identitiesMW is not None else None,
                               identitiesEN.tolist() if identitiesEN is not None else None]

            img = draw_multiple_boxes(bbox_list, mapping_objects, identities_list, [classesWN, classesMSW, classesNS, classesWN2, classesMW, classesEN ], cam_id_list)

            if frame == 3:
                print('here')
            intersecting_bboxes = find_intersections(bbox_list, mapping_objects, classes_list, cam_id_list)

            bbox_all_list = map_bboxes(bbox_list, mapping_objects, cam_id_list, classes_list)
            intersected_bboxes, bbox_xyah = compute_multiple_intersection_bboxes(intersecting_bboxes, bbox_all_list)
            img2 = draw_bboxes(intersected_bboxes, VIDEOFRAME)


            #################################### KALMAN FILTERING ######################################################
            if frame == 3:
                filter_list, mean_list, covariance_list = InitKalmanTracker(bbox_xyah)
                filter_list, mean_list, covariance_list = predictKalmanTracker(filter_list, mean_list, covariance_list)
            elif frame > 3:
                filter_list, mean_list, covariance_list = updateKalmanTracker(filter_list, mean_list, covariance_list)

                # Draw filter outputs
                drawFilterOutput(mean_list, img2)

                filter_list, mean_list, covariance_list = predictKalmanTracker(filter_list, mean_list, covariance_list)

            ######################################## SHOW RESULTS ######################################################
            cv2.imshow('overview', img)
            cv2.imshow('Overview intersection', img2)


            calculated_frames.append(i)
            if cv2.waitKey(0) == 33:
                continue
            frame += 1
            ret, VIDEOFRAME = CAP.read()
            VIDEOFRAME = cv2.resize(VIDEOFRAME, (1788, 1069))
if __name__ == '__main__':
    q = queue.Queue()
    q2 = queue.Queue()
    q3 = queue.Queue()
    q4 = queue.Queue()
    q5 = queue.Queue()
    q6 = queue.Queue()

    # Producers
    threadCamWN = threading.Thread(target=trackerCamWN, args=('videos/VideoWN.mkv',), daemon=True).start() #1
    threadCamMSW = threading.Thread(target=trackerCamMSW, args=('videos/VideoMSW.mkv',), daemon=True).start() # 2
    threadCamNS = threading.Thread(target=trackerCamNS, args=('videos/VideoNS.mkv',), daemon=True).start() #3
    threadCamWN2 = threading.Thread(target=trackerCamWN2, args=('videos/VideoWN2.mkv',), daemon=True).start() #4
    threadCamMW = threading.Thread(target=trackerCamMW, args=('videos/VideoMW.mkv',), daemon=True).start() #5
    threadCamEN = threading.Thread(target=trackerCamEN, args=('videos/VideoEN.mkv',), daemon=True).start() #6

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()