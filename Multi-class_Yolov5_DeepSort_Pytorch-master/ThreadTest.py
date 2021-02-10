import threading, queue
import cv2

import track
from triangulation import triangulate, meterToPixel, euclideanDistance
from Mapping.mapper import arrayToInt

cameraPosWest = meterToPixel(-10.437, 8.9146)
cameraPosMiddle = meterToPixel(24.168, 17.781)

PLANE = cv2.imread('Mapping/plane.png')

def trackerCameraWest(path):
    camera = 'CameraWest'
    q.put(track.run(path, camera))

def trackerCameraSouthWest(path):
    q2.put(track.run(path))

def trackerCameraMiddleFacingWestSouth(path):
    camera = 'CameraMiddelFacingWestSouth'
    q3.put(track.run(path, camera))


def consumer():
    itemCamWest = q.get()
    #itemCamSouthWest = q2.get()
    itemCamMiddle = q3.get()
    print('here')
    #
    for pointsWest, pointsMiddle in zip(itemCamWest, itemCamMiddle):
        triPoints = triangulate([pointsWest, pointsMiddle], [cameraPosWest, cameraPosMiddle])

        for point in triPoints:
            x, y = round(point[0]), round(point[1])
            p = (x,y)

            cv2.circle(PLANE, (x,y),2,(0,255,0),2)
            cv2.imshow('Triangulation', PLANE)

       # testFunc(i)
    """for outputWest, outputSouthWest, outputMiddleWestSouth in zip(itemCamWest, itemCamSouthWest, itemCamMiddleFacingWestSouth):
        for i,j,k in zip(outputWest,outputSouthWest,outputMiddleWestSouth):
            print('i ', i, '\n')
            print('j ', j, '\n')
            print('k ', k, '\n')"""

if __name__ == '__main__':
    q = queue.Queue()
    #q2 = queue.Queue()
    q3 = queue.Queue()

    # Produce
    threadCameraWest = threading.Thread(target=trackerCameraWest, args=('CameraWest.mkv',)).start()
    #threadCameraSouthWest = threading.Thread(target=trackerCameraSouthWest, args=('CameraSouthWest.mkv',)).start()
    threadCameraMiddleFacingWestSouth = threading.Thread(target=trackerCameraMiddleFacingWestSouth, args=('CameraMiddelFacingWestSouth.mkv',)).start()

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()



