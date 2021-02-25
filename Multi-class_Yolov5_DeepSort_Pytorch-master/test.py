from Mapping.mapper import Mapper

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plane = cv.imread('Mapping/plane.png')
cap = cv.VideoCapture('videos/VideoWN.mkv')
ret, camera = cap.read()
ret, camera = cap.read()

pts_src = np.array([[535,150],[720,823],[1215,385],[1561,756],[1495,209],[1835,609],[1434,56]])
pts_dst = np.array([[149,484],[335,963],[612,706 ],[752,937 ],[821,565 ],[919,859],[821,404]])


x = 1434
y = 56
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