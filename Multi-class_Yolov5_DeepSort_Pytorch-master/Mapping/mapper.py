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

from helperFunctions import arrayToInt

import cv2 as cv

class Mapper:
    def __init__(self, planarview, pts_src, pts_dst):
        self.planarView = planarview
        self.H, _ = cv.findHomography(pts_src,pts_dst)

    def getPoint(self):
        midpoint
        # Test
        p = cv.perspectiveTransform(point, self.H)
        x, y = arrayToInt(p)
        return x, y

    def computeMidpoint(self, x1,x2,y1,y2):
        return ((x1 + x2) / 2, (y1 + y2) / 2)
