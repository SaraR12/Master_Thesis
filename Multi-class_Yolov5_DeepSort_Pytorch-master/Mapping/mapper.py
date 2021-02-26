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

from homographies import getKeypoints
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

        p1 = np.array([[x1,y1]], dtype='float32')
        p2 = np.array([[x2, y2]], dtype='float32')

        p1 = np.array([p1])
        p2 = np.array([p2])

        x1m, y1m = self.getPoint(p1)
        x2m, y2m = self.getPoint(p2)


        mappedPoint = self.getPoint(midpoint)
        #mappedImg = cv.circle(self.planarView, mappedPoint, 3, markerColor, 3)
        mappedImg = cv.rectangle(self.planarView, (x1m, y1m), (x2m, y2m), (0,0,255),2)
        return mappedImg, mappedPoint

    def mapBoundingBoxPoints(self,x1,x2,y1,y2, color):
        p1 = np.array([[x1, y1]], dtype='float32')
        p1 = np.array([p1])
        p2 = np.array([[x2, y2]], dtype='float32')
        p2 = np.array([p2])
        x1m, y1m = self.getPoint(p1)
        x2m, y2m = self.getPoint(p2)
        return x1m, x2m, y1m, y2m, color

