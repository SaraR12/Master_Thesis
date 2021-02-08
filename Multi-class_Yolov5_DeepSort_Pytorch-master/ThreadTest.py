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


def consumer():
    item = q.get()
    item2 = q2.get()
    for x, x2 in zip(item,item2):
        for i,i2 in zip(x,x2):
            print(i,i2)

if __name__ == '__main__':

    q = queue.Queue()
    q2 = queue.Queue()
    thread1 = threading.Thread(target=trackerCameraWest, args=('vid4.mkv',)).start()
    thread2 = threading.Thread(target=trackerCameraSouthWest, args=('vid6.mkv',)).start()
    consumerThread = threading.Thread(target=consumer).start()



