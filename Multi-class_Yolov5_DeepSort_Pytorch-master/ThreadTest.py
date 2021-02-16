import threading, queue
import cv2
import time

import track
from triangulation import triangulate, meterToPixel, euclideanDistance
from Mapping.mapper import *
from Mapping.bbox_intersection import *
from homographies import *

cameraPosWest = meterToPixel(-10.437, 8.9146)
cameraPosMiddle = meterToPixel(24.168, 17.781)

PLANE = cv2.imread('Mapping/plane.png')
q1List = []
q2List = []
q3List = []
q4List = []
q5List = []
q6List = []

def runTracker(path, camera, queue=None):
    camera = 'WN'
    out = track.run(path, camera,queue)
    for i in out:
        q1List.append(i)
    print(q1List)
def trackerCamWN(path):
    camera = 'WN'
    out = track.run(path, camera)
    for i in out:
        q1List.append([i])
    #q.put(track.run(path, camera), block=True)

def trackerCamMSW(path):
    camera = 'MSW'
    out = track.run(path, camera)
    for i in out:
        q2List.append([i])
    #q2.put(track.run(path, camera), block=True)

def trackerCamNS(path):
    camera = 'NS'
    out = track.run(path, camera)
    for i in out:
        q3List.append([i])
    #q3.put(track.run(path, camera), block=True)

def trackerCamME(path):
    camera = 'ME'
    out = track.run(path, camera)
    for i in out:
        q4List.append([i])
    #q4.put(track.run(path, camera), block=True)

def trackerCamMW(path):
    camera = 'MW'
    out = track.run(path, camera)
    for i in out:
        q5List.append([i])
    #q5.put(track.run(path, camera), block=True)

def trackerCamEN(path):
    camera = 'EN'
    out = track.run(path, camera)
    for i in out:
        q6List.append([i])
    #q6.put(track.run(path, camera), block=True)


def consumer():
    pts_src, pts_dst = getKeypoints('WN')
    mapObj1 = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("MSW")
    mapObj2 = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("NS")
    mapObj3 = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("ME")
    mapObj4 = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("MW")
    mapObj5 = Mapper(PLANE, pts_src, pts_dst)

    pts_src, pts_dst = getKeypoints("EN")
    mapObj6 = Mapper(PLANE, pts_src, pts_dst)

    mapping_objects = [mapObj1, mapObj2, mapObj3, mapObj4, mapObj5, mapObj6]

    calculated_frames = []

    while True:
        i = min(len(q1List), len(q2List), len(q3List), len(q4List), len(q5List), len(q6List))
        print(i)
        """itemCamWN = q.get()
        itemCamMSW = q2.get()
        itemCamNS = q3.get()
        itemCamME = q4.get()
        itemCamMW = q5.get()
        itemCamEN = q6.get()
        print('qsize ', q.qsize())
        
        print('here')"""
        #print(len(q1List), len(q2List), len(q3List), len(q4List), len(q5List), len(q6List))
        #print(q1List)
        #if (i > 0) and (i not in calculated_frames):
        if all([q1List,q2List,q3List,q4List,q5List,q6List]):

            q1Value = q1List.pop(0)
            if q1Value[0] is not None:
                bbox_xyxy1 = q1Value[0][:][:,:4]
                identities1 = q1Value[0][:][:,4]
                frame1 = q1Value[0][:][0,7]
                print('Frame at vid 1:', frame1)
            else:
                bbox_xyxy1 = None
                identities1 = None

            q2Value = q2List.pop(0)
            if q2Value[0] is not None:
                bbox_xyxy2 = q2Value[0][:][:,:4]
                identities2 = q2Value[0][:][:,4]
                frame2 = q2Value[0][:][0,7]
                print('Frame at vid 2:', frame2)
            else:
                bbox_xyxy2 = None
                identities2 = None
                #print('Thread 2 reporting None')

            q3Value = q3List.pop(0)
            if q3Value[0] is not None:
                bbox_xyxy3 = q3Value[0][:][:,:4]
                identities3 = q3Value[0][:][:,4]
                frame3 = q3Value[0][:][0,7]
                print('Frame at vid 3:', frame3)
            else:
                bbox_xyxy3 = None
                identities3 = None

            q4Value = q4List.pop(0)
            if q3Value[0] is not None:
                bbox_xyxy4 = q4Value[0][:][:,:4]
                identities4 = q4Value[0][:][:,4]
                frame4 = q4Value[0][:][0,7]
                print('Frame at vid 4:', frame4)
            else:
                bbox_xyxy4 = None
                identities4 = None

            q5Value = q5List.pop(0)
            if q5Value[0] is not None:
                bbox_xyxy5 = q5Value[0][:][:,:4]
                identities5 = q5Value[0][:][:,4]
                frame5 = q5Value[0][:][0,7]
                print('Frame at vid 5:', frame5)
            else:
                bbox_xyxy5 = None
                identities5 = None

            q6Value = q6List.pop(0)
            if q6Value[0] is not None:
                bbox_xyxy6 = q6Value[0][:][:,:4]
                identities6 = q6Value[0][:][:,4]
                frame6 = q6Value[0][:][0,7]
                print('Frame at vid 6:', frame6)
            else:
                #print('Thread 6 reporting None')
                bbox_xyxy6 = None
                identities6 = None


                """if all(SYNC):
                    print('All processes executed')
                    print(q2.qsize())
                    time.sleep(2)
                    continue"""
            bbox_list = [bbox_xyxy1 if bbox_xyxy1 is not None else [],
                         bbox_xyxy2 if bbox_xyxy2 is not None else [],
                         bbox_xyxy3 if bbox_xyxy3 is not None else [],
                         bbox_xyxy4 if bbox_xyxy4 is not None else [],
                         bbox_xyxy5 if bbox_xyxy5 is not None else [],
                         bbox_xyxy6 if bbox_xyxy6 is not None else []]

            intersected_bboxes = iou_bboxes(bbox_list, mapping_objects)

            identities_list = [identities1, identities2, identities3, identities4, identities5, identities6]
            #bbox_list = [bbox_xyxy1]
            #identities_list = [identities1]
            #mapping_objects = [mapObj1]
            img = draw_multiple_boxes(bbox_list, mapping_objects, identities_list)
            img2 = draw_bboxes(intersected_bboxes)

            cv2.imshow('overview', img)
            cv2.imshow('Overview intersection', img2)
            calculated_frames.append(i)
            if cv2.waitKey(0) == 33:
                continue


if __name__ == '__main__':
    q = queue.Queue()
    q2 = queue.Queue()
    q3 = queue.Queue()
    q4 = queue.Queue()
    q5 = queue.Queue()
    q6 = queue.Queue()


    # Producers
    threadCamWN = threading.Thread(target=trackerCamWN, args=('videos/VideoNW.mkv',), daemon=True).start()
    threadCamMSW = threading.Thread(target=trackerCamMSW, args=('videos/VideoMSW.mkv',), daemon=True).start()
    threadCamNS = threading.Thread(target=trackerCamNS, args=('videos/VideoNS.mkv',), daemon=True).start()
    threadCamME = threading.Thread(target=trackerCamME, args=('videos/VideoME.mkv',), daemon=True).start()
    threadCamMW = threading.Thread(target=trackerCamMW, args=('videos/VideoMW.mkv',), daemon=True).start()
    threadCamEN = threading.Thread(target=trackerCamEN, args=('videos/VideoEN.mkv',), daemon=True).start()


    # Consumer
    consumerThread = threading.Thread(target=consumer).start()





