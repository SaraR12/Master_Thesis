from shapely.geometry import LineString
import math as m
import numpy as np

""" 
Part of Master Thesis 'Indoor Tracking using a Central Camera System' at Chalmers University of Technology, conducted
at Sigma Technology Insights 2021.

Authors:
Jonas Lindberg
Sara Roth

"""

def euclideanDistance(p1,p2):
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    return m.sqrt((x1-x2)**2 + (y1-y2)**2)

def meterToPixel(xm,ym):
    scalex = 1788 / 50
    x = xm * scalex
    scaley = 1069 / 30
    y = (30 - ym) * scaley
    return (round(x), round(y))

def pixelToMeter(xp, yp):
    scalex = 50 / 1788
    x = xp * scalex

    scaley = 30 / 1069
    y = (1069 - yp) * scaley
    return (x,y)

def triangulate(projectedPoints, cameraPositions):
    triangulatedPoints = []

    cam1 = cameraPositions[0]
    cam2 = cameraPositions[1]
    proj1 = projectedPoints[0]
    proj2 = projectedPoints[1]
    for p1 in proj1:
        line1 = LineString([cam1, p1])
        for p2 in proj2:
            line2 = LineString([cam2, p2])
            intersection = np.array(line1.intersection(line2))
            # if there is an intersection, save it

            if not intersection.size == 0:
                triangulatedPoints.append(intersection)

    return triangulatedPoints


