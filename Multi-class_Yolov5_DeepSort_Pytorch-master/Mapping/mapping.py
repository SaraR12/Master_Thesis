import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
Mapper Class:
Maps a point from one perspective into another.
Init function takes an image in the desired perspective (e.g. a planar map).

Mapper.map()


"""
class Mapper:
    def __init__(self, planarView):
        self.planarView = planarView

        # Initialize BRISK algorithm
        self.brisk = cv.BRISK_create()

        # Initialize matcher (BruteForce)
        self.BFMatcher = cv.BFMatcher(normType = cv.NORM_HAMMING,
                         crossCheck = True)

    def SIFTDetectAndMatch(self, cameraView, planarView):
        sift = cv.xfeatures2d_SIFT.create()
        return None

    def matchKeypoints(self, cameraView, planarView):
        # Compute keypoints and descriptors
        kp1, des1 = self.brisk.detectAndCompute(cameraView, None)
        kp2, des2 = self.brisk.detectAndCompute(planarView, None)
        sift = cv.xfeatures2d_SIFT.create()
        # Match keypoints
        matches = self.BFMatcher.match(queryDescriptors=des1,
                                  trainDescriptors=des2)

        matches = sorted(matches, key=lambda x: x.distance)
        output = cv.drawMatches(img1=cameraView,
                                keypoints1=kp1,
                                img2=planarView,
                                keypoints2=kp2,
                                matches1to2=matches[15:],
                                outImg=None,
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(output)
        plt.show()
        return None

    def estimateHomography(self, pts_src, pts_dst):
        H, _ = cv.findHomography(pts_src, pts_dst)
        return H

    def map(self, H, point):
        return cv.perspectiveTransform(point, H)
# Read perspective (camera) image
camera = cv.imread('SE.png')

# Read 2D plane image
plane = cv.imread('plane.png', 1)

plane = cv.cvtColor(plane, cv.COLOR_BGR2RGB)
camera = cv.cvtColor(camera, cv.COLOR_BGR2RGB)

CAMERA = 'CenterNorth'
#CAMERA = 'CenterNorth'

if CAMERA == 'NorthEast':
    pts_src = np.array([[972,858],[385,676],[153,606],[1399,572],[1517,495],[1600,441],[1638,416]])
    pts_dst = np.array([[247,794],[251,372],[251,107],[759,790],[1036,789],[1310,791],[1470,790]])

    a = np.array([[950, 956]], dtype="float32")
    a = np.array([a])

    b = np.array([[1499, 565]], dtype='float32')
    b = np.array([b])

    cv.circle(plane, (177, 833), 5, (0, 255, 0), 3)
    cv.circle(camera, (950, 956), 3, (0, 255, 0), 3)

    cv.circle(plane, (812, 842), 5, (255, 0, 0), 3)
    cv.circle(camera, (1499, 565), 3, (255, 0, 0), 3)

if CAMERA == 'CenterNorth':
    pts_src = np.array([[1024,925],[1380,626],[615,619],[414,471],[219,332],[745,718],[188,308],[203,319]])
    pts_dst = np.array([[1681,866],[1680,569],[1376,865],[1069,866],[499,869],[1503,865],[343,869],[421,869]])

    a = np.array([[79,302]], dtype='float32')
    a = np.array([a])

    b = np.array([[345,513]], dtype='float32')
    b = np.array([b])

    cv.circle(camera,(79,302), 3, (0,255,0), 3)
    cv.circle(plane, (309,931), 5, (0,255,0),3)

    cv.circle(camera, (345,513),3,(255,0,0),3)
    cv.circle(plane, (1141,909), 5,(255,0,0),3)


h, status = cv.findHomography(pts_src, pts_dst)
#cameraWarped = cv.warpPerspective(camera,h,(1920,1080))

points = cv.perspectiveTransform(b,h)
pointsOut = cv.perspectiveTransform(a, h)
print('a = ', pointsOut)
print('b = ', points)


#camera = cv.warpPerspective(camera, h, (1920,1080))


#cv.imshow('1',camera)
#cv.imshow('2',plane)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Test')
ax1.imshow(camera)
ax2.imshow(plane)
plt.show()
#cv.waitKey(0)
