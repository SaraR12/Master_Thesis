import threading, queue
import cv2

import track
from triangulation import triangulate, meterToPixel, euclideanDistance
from Mapping.mapper import *
from Mapping.bbox_intersection import *
from homographies import *

cameraPosWest = meterToPixel(-10.437, 8.9146)
cameraPosMiddle = meterToPixel(24.168, 17.781)

PLANE = cv2.imread('Mapping/plane.png')

def trackerCamWN(path):
    camera = 'WN'
    q.put(track.run(path, camera))

def trackerCamMSW(path):
    camera = 'MSW'
    q2.put(track.run(path, camera))

def trackerCamNS(path):
    camera = 'NS'
    q3.put(track.run(path, camera))

def trackerCamME(path):
    camera = 'ME'
    q4.put(track.run(path, camera))

def trackerCamMW(path):
    camera = 'MW'
    q5.put(track.run(path, camera))

def trackerCamEN(path):
    camera = 'EN'
    q6.put(track.run(path, camera))


def consumer():
    itemCamWN = q.get()
    itemCamMSW = q2.get()
    itemCamNS = q3.get()
    itemCamME = q4.get()
    itemCamMW = q5.get()
    itemCamEN = q6.get()


    print('here')
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

    for out1,out2, out3, out4, out5, out6 in zip(itemCamWN, itemCamMSW, itemCamNS, itemCamME, itemCamMW, itemCamEN):
        bbox_xyxy1 = out1[:, :4]
        identities1 = out1[:, 4]

        bbox_xyxy2 = out2[:, :4]
        identities2 = out2[:, 4]

        bbox_xyxy3 = out3[:, :4]
        identities3 = out3[:, 4]

        bbox_xyxy4 = out4[:, :4]
        identities4 = out4[:, 4]

        bbox_xyxy5 = out5[:, :4]
        identities5 = out5[:, 4]

        bbox_xyxy6 = out6[:, :4]
        identities6 = out6[:, 4]


        bbox_list = [bbox_xyxy1, bbox_xyxy2, bbox_xyxy3, bbox_xyxy4, bbox_xyxy5, bbox_xyxy6]

        intersected_bboxes = iou_bboxes(bbox_list, mapping_objects)

        identities_list = [identities1, identities2, identities3, identities4, identities5, identities6]
        #bbox_list = [bbox_xyxy1]
        #identities_list = [identities1]
        #mapping_objects = [mapObj1]
        img = draw_multiple_boxes(bbox_list, mapping_objects, identities_list)
        img2 = draw_bboxes(intersected_bboxes)
        cv2.imshow('overview', img)
        cv2.imshow('Overview intersection', img2)
        if cv2.waitKey(0) == 3:
            continue



if __name__ == '__main__':
    q = queue.Queue()
    q2 = queue.Queue()
    q3 = queue.Queue()
    q4 = queue.Queue()
    q5 = queue.Queue()
    q6 = queue.Queue()

    # Produce
    threadCamWN = threading.Thread(target=trackerCamWN, args=('videos/VideoWN.mkv',)).start()
    threadCamMSW = threading.Thread(target=trackerCamMSW, args=('videos/VideoMSW.mkv',)).start()
    threadCamNS = threading.Thread(target=trackerCamNS, args=('videos/VideoNS.mkv',)).start()
    threadCamME = threading.Thread(target=trackerCamME, args=('videos/VideoME.mkv',)).start()
    threadCamMW = threading.Thread(target=trackerCamMW, args=('videos/VideoMW.mkv',)).start()
    threadCamEN = threading.Thread(target=trackerCamEN, args=('videos/VideoEN.mkv',)).start()

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()



