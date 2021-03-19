from shapely.geometry import LineString
import math as m
import numpy as np

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


"""cameraPos1 = meterToPixel(-10.437, 8.9146)

projPoint1 = (92,723)


cameraPos2 = meterToPixel(24.168, 17.781)

projPoint2 = (40,740)


truePos = meterToPixel(2.4685, 9.682)
#print((2.4685, 9.682))

rms = m.sqrt((2.4685-2.371886054303795)**2+(9.682 - 9.697731229596354)**2)
#print("Triangulation error (m): ",rms)

camPos = [cameraPos1, cameraPos2]
projPoints = [[projPoint1], [projPoint2]]
triangulate(projPoints, camPos)"""


