from Mapping.mapper import Mapper

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plane = cv.imread('Mapping/plane.png')
cap = cv.VideoCapture('videos/VideoM.mkv')
ret, camera = cap.read()
ret, camera = cap.read()

pts_src = np.array([[557,738],[1548,405],[1037,113],[1321,14],[618,57],[425,521],[524,278]])
pts_dst = np.array([[146,488],[332,966],[608,709],[748,941],[683,399],[257,396],[433,396]])
[[], [], [], [], [], [], []]
x = 481
y = 396
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