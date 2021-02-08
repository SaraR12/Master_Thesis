import multiprocessing as mp # https://www.kth.se/blogs/pdc/2019/02/parallel-programming-in-python-multiprocessing-part-1/
import os
import torch
import torch.backends.cudnn as cudnn
import threading, queue

from track import runTracker

def trackerCameraWest(path):
    q.put(runTracker(path))


def trackerCameraSouthWest(path):
    q2.put(runTracker(path))

def trackerCameraMiddleFacingWestSouth(path):
    q3.put(runTracker(path))


def consumer():
    itemCamWest = q.get()
    itemCamSouthWest = q2.get()
    itemCamMiddleFacingWestSouth = q3.get()

    for outputWest, outputSouthWest, outputMiddleWestSouth in zip(itemCamWest, itemCamSouthWest, itemCamMiddleFacingWestSouth):


    for x, x2 in zip(item,item2):
        for i,i2 in zip(x,x2):
            print(i,i2)

if __name__ == '__main__':

    q = queue.Queue()
    q2 = queue.Queue()
    q3 = queue.Queue()

    # Produce
    threadCameraWest = threading.Thread(target=trackerCameraWest, args=('CameraWest.mkv',)).start()
    threadCameraSouthWest = threading.Thread(target=trackerCameraSouthWest, args=('CameraSouthWest.mkv',)).start()
    threadCameraMiddleFacingWestSouth = threading.Thread(target=trackerCameraSouthWest, args=('CameraMiddleFacingWestSouth.mkv',)).start()

    # Consumer
    consumerThread = threading.Thread(target=consumer).start()



