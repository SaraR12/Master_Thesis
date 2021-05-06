from cv2 import findHomography, perspectiveTransform, RANSAC, rectangle
from numpy import array

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

def arrayToInt(p):
    x = int(p[:,:,0])
    y = int(p[:,:,1])
    return x,y

class Mapper:
    def __init__(self, planarview, pts_src, pts_dst):
        self.planarView = planarview
        self.H, _ = findHomography(pts_src,pts_dst, RANSAC, 5.0)

    def getPoint(self, point):
        p = perspectiveTransform(point, self.H)
        x, y = arrayToInt(p)
        return x, y

    def mapFromBoundingBox(self, x1,x2,y1,y2, markerColor):
        midx = round((x1 + x2) / 2)
        midy = round((y1 + y2) / 2)

        midx = int(midx)
        midy = int(midy)

        midpoint = array([[midx,midy]], dtype='float32')
        midpoint = array([midpoint])

        p1 = array([[x1,y1]], dtype='float32')
        p2 = array([[x2, y2]], dtype='float32')

        p1 = array([p1])
        p2 = array([p2])

        x1m, y1m = self.getPoint(p1)
        x2m, y2m = self.getPoint(p2)


        mappedPoint = self.getPoint(midpoint)
        mappedImg = rectangle(self.planarView, (x1m, y1m), (x2m, y2m), (0,0,255),2)
        return mappedImg, mappedPoint

    def mapBoundingBoxPoints(self,x1,x2,y1,y2, color):
        p1 = array([[x1, y1]], dtype='float32')
        p1 = array([p1])
        p2 = array([[x2, y2]], dtype='float32')
        p2 = array([p2])
        x1m, y1m = self.getPoint(p1)
        x2m, y2m = self.getPoint(p2)
        return x1m, x2m, y1m, y2m, color

