from shapely.geometry import LineString

class Triangulator:
    def __init__(self, cameraPos):
        self.camPos = cameraPos

    def triangulate(self, projPoints1,cameraPos1, projPoints2, cameraPos2):
        triangulatedPoints = []
        for p1 in projPoints1:
            for p2 in projPoints2:
                line1 = LineString([(cameraPos1), (p1)])

