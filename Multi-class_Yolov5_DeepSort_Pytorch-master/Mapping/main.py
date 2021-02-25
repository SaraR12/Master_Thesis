from mapper import Mapper

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plane = cv.imread('plane.png')
cap = cv.VideoCapture('Multi-class_Yolov5_DeepSort_Pytorch-master/videos/VideoWN.mkv')
ret, camera = cap.read()

pts_src = np.array([[535,150],[720,823],[1215,385],[1561,756],[1495,209],[1832,607],[1434,56]])
pts_dst = np.array([[149,485],[335,963],[612,706],[752,937],[321,565],[919,859],[821,404]])


x = 535
y = 150
mapper = Mapper(plane, pts_src, pts_dst)
point = np.array([[535,150]], dtype='float32')
point = np.array([point])

xtilde, ytilde = mapper.getPoint(point)

cv.circle(camera, (x, y), 7, (0, 0, 255), 3)
cv.circle(plane, (xtilde, ytilde), 7, (0, 0, 255), 3)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Mapping test')
camera = cv.cvtColor(camera, cv.COLOR_BGR2RGB)
plane = cv.cvtColor(plane, cv.COLOR_BGR2RGB)
ax1.imshow(camera)
ax2.imshow(plane)

plt.show()