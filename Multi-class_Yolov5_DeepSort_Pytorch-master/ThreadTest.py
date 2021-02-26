import threading, queue, time

import track
from Mapping.mapper import *
from Mapping.bbox_intersection import *
from homographies import *
from KalmanTracker import *
import trackNoDeepSort
from deep_sort.deep_sort.deep_sort import *
from deep_sort.deep_sort.sort.tracker import *
from deep_sort.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.sort.detection import Detection
import torch

qWNList = []
qMSWList = []
qNSList = []
qMList = []
qENList = []
qMEList = []


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

    mapping_objects = [mapObjWN, mapObjMSW, mapObjNS, mapObjM, mapObjEN, mapObjME]

    ########## DeepSORT #########

    DeepSortObj = DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7')

    max_cosine_distance = 0.2 # max_dist Might be changed later
    nn_budget = 100
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    #############################
    calculated_frames = []
    i = 0
    frame = 1
    while i < 300:
        i = min(len(qWNList), len(qMSWList), len(qNSList), len(qMList), len(qENList), len(qMEList))

        classesWN, classesMSW, classesNS, classesM, classesEN, classesME = np.array([]), np.array([]), \
                                                                                        np.array([]),np.array([]), np.array([]), np.array([])
        confWN, confMSW, confNS, confM, confEN, confME = np.array([]), np.array([]), \
                                                                                        np.array([]),np.array([]), np.array([]), np.array([])

        if all([qWNList,qMSWList,qNSList,qMList,qENList, qMEList]):
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
                identitiesM = np.ones(len(qMValue[0][:][:,4])) * 5
                classesM = qMValue[0][:][:,5]
                confM = qMValue[0][:][:,6]
            else:
                bbox_xyxyM = None
                identitiesM = None

            qENValue = qENList.pop(0)
            if qENValue[0] is not None:
                bbox_xyxyEN = qENValue[0][:][:,:4]
                identitiesEN = np.ones(len(qENValue[0][:][:,4])) * 6
                classesEN = qENValue[0][:][:,5]
                confEN = qENValue[0][:][:,6]
            else:
                bbox_xyxyEN = None
                identitiesEN = None

            qMEValue = qMEList.pop(0)
            if qMEValue[0] is not None:
                bbox_xyxyME = qMEValue[0][:][:,:4]
                identitiesME = np.ones(len(qMEValue[0][:][:,4])) * 7
                classesME = qMEValue[0][:][:,5]
                confME = qMEValue[0][:][:,6]
            else:
                bbox_xyxyME = None
                identitiesME = None


            bbox_list = [bbox_xyxyWN if bbox_xyxyWN is not None else [],
                         bbox_xyxyMSW if bbox_xyxyMSW is not None else [],
                         bbox_xyxyNS if bbox_xyxyNS is not None else [],
                         bbox_xyxyM if bbox_xyxyM is not None else [],
                         bbox_xyxyEN if bbox_xyxyEN is not None else [],
                         bbox_xyxyME if bbox_xyxyME is not None else []]

            cam_id_list = [[np.ones(len(bbox_xyxyWN if bbox_xyxyWN is not None else []))[:].tolist()] +
                           [(np.ones(len(bbox_xyxyMSW if bbox_xyxyMSW is not None else []))*2)[:].tolist()] +
                           [(np.ones(len(bbox_xyxyNS if bbox_xyxyNS is not None else []))*3)[:].tolist()] +
                           [(np.ones(len(bbox_xyxyM if bbox_xyxyM is not None else []))*4)[:].tolist()] +
                           [(np.ones(len(bbox_xyxyEN if bbox_xyxyEN is not None else []))*5)[:].tolist()] +
                           [(np.ones(len(bbox_xyxyME if bbox_xyxyME is not None else []))*6)[:].tolist()]
                           ][0]

            classes_list = [classesWN.tolist() + classesMSW.tolist() + classesNS.tolist() +
                            classesM.tolist() + classesEN.tolist() + classesME.tolist()][0]
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
                               identitiesME.tolist() if identitiesME is not None else None]

            conf_list = [confWN.tolist() + confMSW.tolist() + confNS.tolist() + confM.tolist() + confEN.tolist() + confME.tolist()][0]

            img = draw_multiple_boxes(bbox_list, mapping_objects, identities_list, [classesWN, classesMSW, classesNS, classesM, classesEN, classesME], cam_id_list, VIDEOFRAME)
            intersecting_bboxes, intersecting_classes_list = find_intersections(bbox_list, mapping_objects, classes_list, cam_id_list)
            print(intersecting_bboxes)
            bbox_all_list = map_bboxes(bbox_list, mapping_objects, cam_id_list, classes_list)
            intersected_bboxes, bbox_xyah, bbox_xywh = compute_multiple_intersection_bboxes(intersecting_bboxes, bbox_all_list)



            """# Pass detections to deepsort
            xywhs = torch.Tensor(bbox_xywh)
            detections = [Detection(bbox_xywh[i], 1, [], cls) for i, cls in enumerate(intersecting_classes_list)]
            tracker.predict()
            tracker.update(detections)
            print(tracker.tracks)

            outputs = []
            now_time = datetime.datetime.now()
            for now_line, track in enumerate(tracker.tracks):
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1, y1, x2, y2 = tlwh_to_xyxy(box)
                track_id = track.track_id
                cls2 = track.cls
                score2 = int(track.score * 100)
                start_time = track.start_time
                stay = int((now_time - start_time).total_seconds())
                outputs.append(np.array([x1, y1, x2, y2, track_id, cls2, score2, stay], dtype=np.int))
            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)
                print(outputs.shape)
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, 4]
                print(identities  )
                img2 = draw_bboxes(bbox_xyxy, VIDEOFRAME, identities)
                cv2.imshow('Overview intersection', img2)"""
            """
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(np.ones(len(xywhs)))
            clses = torch.Tensor(intersecting_classes_list)

            outputs = DeepSortObj.update(xywhs, confss, clses, PLANE)
            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)
                print(outputs.shape)
                 bbox_xyxy = outputs[:, :4]
                identities = outputs[:, 4]
                print(identities)
                img2 = draw_bboxes(bbox_xyxy, VIDEOFRAME, identities)
                cv2.imshow('Overview intersection', img2)"""
            #t3 = time_synchronized()
            if False:
                img2 = draw_bboxes(intersected_bboxes, VIDEOFRAME)
                #################################### KALMAN FILTERING ######################################################

                if frame == 3:
                    filter_list, mean_list, covariance_list = InitKalmanTracker(bbox_xyah)
                    #filter_list, mean_list, covariance_list = predictKalmanTracker(filter_list, mean_list, covariance_list)
                elif frame > 3:
                    print(frame)
                    filter_list, mean_list, covariance_list = predictKalmanTracker(filter_list, mean_list, covariance_list)
                    bbox_xyah = association(filter_list, mean_list, covariance_list, bbox_xyah)
                    #mean_list, covariance_list = projectKalmanTracker(filter_list, mean_list, covariance_list)
                    filter_list, mean_list, covariance_list = updateKalmanTracker(filter_list, mean_list, covariance_list, bbox_xyah)

                    # Draw filter outputs
                    drawFilterOutput(mean_list, img2)


                    cv2.imshow('Overview intersection', img2)
                ######################################## SHOW RESULTS ######################################################
            cv2.imshow('overview', img)



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
    threadCamMW = threading.Thread(target=trackerCamM, args=('videos/VideoM.mkv',), daemon=True).start() #5
    threadCamEN = threading.Thread(target=trackerCamEN, args=('videos/VideoEN.mkv',), daemon=True).start() #6
    threadCamME = threading.Thread(target=trackerCamME, args=('videos/VideoME.mkv',), daemon=True).start() # 7

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()