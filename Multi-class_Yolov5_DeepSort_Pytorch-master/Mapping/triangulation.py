class Triangulator:
    def __init__(self, cameraPositions):
        self.cameraPos = cameraPositions

    def triangulate(self, detections):
        for cam, detect in zip(self.cameraPos, detections):
            pass

    def getLine(self, camXY, detectionXY):
        pass