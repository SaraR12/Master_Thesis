from Mapping.mapper import Mapper
from homographies import getKeypoints

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

""" Testfile to compare the mapped points to the points in camera view

Part of Master Thesis 'Indoor Tracking using a Central Camera System' at Chalmers University of Technology, conducted
at Sigma Technology Insights 2021.

Authors:
Jonas Lindberg
Sara Roth
"""

# Create frames from the video
plane = cv.imread('Mapping/plane.png')
cap = cv.VideoCapture('videos/VideoM.mkv')
ret, camera = cap.read()

# Homography for the camera we want to test
pts_src, pts_dst = getKeypoints('M')

# The x and y points we want to test
x = 481
y = 396

# Get the mapper object an the points we want to test
mapper = Mapper(plane, pts_src, pts_dst)
point = np.array([[x,y]], dtype='float32')
point = np.array([point])

# Get the mapped pont
xtilde, ytilde = mapper.getPoint(point)
print(xtilde, ytilde)
cv.circle(camera, (x, y), 1, (0, 0, 255), 3)
cv.circle(plane, (xtilde, ytilde), 1, (0, 0, 255), 3)

# Show the results
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Mapping test')

ax1.imshow(camera)
ax2.imshow(plane)
plt.show()