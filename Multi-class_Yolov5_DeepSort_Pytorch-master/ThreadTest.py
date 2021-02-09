import threading, queue

import track
from triangulation import testFunc

def trackerCameraWest(path):
    q.put(track.run(path))

def trackerCameraSouthWest(path):
    q2.put(track.run(path))

def trackerCameraMiddleFacingWestSouth(path):
    q3.put(track.run(path))


def consumer():
    itemCamWest = q.get()
    #itemCamSouthWest = q2.get()
    #itemCamMiddleFacingWestSouth = q3.get()
    #for i, j in zip(itemCamWest, itemCamSouthWest):
        #print(i, "--", j)

    #
    for i in itemCamWest:
        testFunc(i)
    """for outputWest, outputSouthWest, outputMiddleWestSouth in zip(itemCamWest, itemCamSouthWest, itemCamMiddleFacingWestSouth):
        for i,j,k in zip(outputWest,outputSouthWest,outputMiddleWestSouth):
            print('i ', i, '\n')
            print('j ', j, '\n')
            print('k ', k, '\n')"""

if __name__ == '__main__':
    q = queue.Queue()
    q2 = queue.Queue()
    q3 = queue.Queue()

    # Produce
    threadCameraWest = threading.Thread(target=trackerCameraWest, args=('CameraWest.mkv',)).start()
    #threadCameraSouthWest = threading.Thread(target=trackerCameraSouthWest, args=('CameraSouthWest.mkv',)).start()
    #threadCameraMiddleFacingWestSouth = threading.Thread(target=trackerCameraSouthWest, args=('CameraMiddelFacingWestSouth.mkv',)).start()

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()



