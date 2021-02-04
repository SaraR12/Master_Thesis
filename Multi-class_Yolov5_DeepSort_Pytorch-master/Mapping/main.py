from mapper import Mapper
from helperFunctions import *

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plane = cv.imread('plane.png')
camera = cv.imread('SE.png')

pts_src = np.array([[1017,883],[1335,616],[651,611],[533,525],[291,430],[470,477],[1533,492]])
pts_dst = np.array([[1680,864],[1679,569],[1375,864],[1197,864],[822,941],[1068,864],[1708,270]])

mapper = Mapper(plane, pts_src, pts_dst)
point = np.array([[237,393]], dtype='float32')
point = np.array([point])

xtilde, ytilde = mapper.getPoint(point)

x, y = arrayToInt(point)

cv.circle(camera, (x, y), 7, (0, 0, 255), 3)
cv.circle(plane, (xtilde, ytilde), 7, (0, 0, 255), 3)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Mapping test')
camera = cv.cvtColor(camera, cv.COLOR_BGR2RGB)
plane = cv.cvtColor(plane, cv.COLOR_BGR2RGB)
ax1.imshow(camera)
ax2.imshow(plane)

plt.show()