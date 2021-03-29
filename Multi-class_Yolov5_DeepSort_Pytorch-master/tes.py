from track import runTracker
import cv2
path = 'Multi-class_Yolov5_DeepSort_Pytorch-master/vid4.mkv'

im = runTracker(path)
cv2.imshow(im)
cv2.waitKey(0)
