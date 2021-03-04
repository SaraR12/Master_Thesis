from Mapping.mapper import Mapper

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plane = cv.imread('Mapping/plane.png')
cap = cv.VideoCapture('videos/VideoMW.mkv')
ret, camera = cap.read()
ret, camera = cap.read()

pts_src = np.array([[1455,240],[126,276],[865,557],[859,264],[1705,554],[869,964],[1107,557]])
pts_dst = np.array([[1309,400],[564,397],[997,564],[997,403],[1304,564],[997,786],[1127,564]])

x = 1455
y = 240
mapper = Mapper(plane, pts_src, pts_dst)
point = np.array([[x,y]], dtype='float32')
point = np.array([point])

xtilde, ytilde = mapper.getPoint(point)
print(xtilde, ytilde)
cv.circle(camera, (x, y), 1, (0, 0, 255), 3)
cv.circle(plane, (xtilde, ytilde), 1, (0, 0, 255), 3)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Mapping test')

ax1.imshow(camera)
ax2.imshow(plane)

plt.show()