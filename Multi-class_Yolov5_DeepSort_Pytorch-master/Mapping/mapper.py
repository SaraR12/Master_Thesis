"""
        Mapper
- Maps a point from one perspective into another.

Mapper init takes the desired destination perspective (e.g. planar)

Mapper.getPoint takes as input a point in another perspective and returns the projected point in the perspective of the
destination perspective.

Part of Master Thesis 'Indoor Tracking using a Central Camera System' at Chalmers University of Technology, conducted
at Sigma Technology Insights 2021.

Authors:
Jonas Lindberg
Sara Roth
"""

#from helperFunctions import arrayToInt

import cv2 as cv
import numpy as np
def arrayToInt(p):
    x = int(p[:,:,0])
    y = int(p[:,:,1])
    return x,y

class Mapper:
    def __init__(self, planarview, pts_src, pts_dst):
        self.planarView = planarview
        self.H, _ = cv.findHomography(pts_src,pts_dst)

    def getPoint(self, point):
        p = cv.perspectiveTransform(point, self.H)
        x, y = arrayToInt(p)
        return x, y

    def mapFromBoundingBox(self, x1,x2,y1,y2, markerColor):
        midx = round((x1 + x2) / 2)
        midy = round((y1 + y2) / 2)

        midx = int(midx)
        midy = int(midy)

        midpoint = np.array([[midx,midy]], dtype='float32')
        midpoint = np.array([midpoint])


        mappedPoint = self.getPoint(midpoint)
        mappedImg = cv.circle(self.planarView, mappedPoint, 3, markerColor, 3)
        return mappedImg, mappedPoint
