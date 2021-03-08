import cv2
import numpy as np
from mapper import Mapper
from homographies import getKeypoints
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('videos/VideoMSW.mkv')
ret, frame = cap.read()

plane = cv2.imread('Mapping/plane.png')

pts_src, pts_dst = getKeypoints('MSW')

mapper = Mapper(plane,pts_src,pts_dst)

a = np.array([[1267,832]], dtype='float32')
a = np.array([a])

(x, y) = mapper.getPoint(a)

cv2.circle(frame, (1267,832),2,(0,255,0),3)
cv2.circle(plane, (x,y), 2, (0,255,0),3)
print((x,y))

plt.subplot(1,2,1)
plt.imshow(frame)
plt.subplot(1,2,2)
plt.imshow(plane)
plt.show()