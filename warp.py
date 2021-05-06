import cv2 as cv
import numpy as np
from homographies import *
pts_src, pts_dst = getKeypoints('TL')
def drawMatches(img1, img2, pts_src=None, pts_dst=None):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    h = h1 if h1 > h2 else h2
    w = w1 if w1 > w2 else w2

    dh, dw = h - min(h1, h2), w - min(w1, w2)

    if h2 < h1:
        img2 = pad(img2, dh, dw)

    img3 = cv.hconcat([img1, img2])

    for p_s, p_d in zip(pts_src, pts_dst):
        x_s, y_s = p_s
        x_d, y_d = p_d

        #color = (np.random.rand()*255, np.random.rand()*255, np.random.rand()*255)
        color = (173, 203, 248)

        cv.circle(img3,(x_s, y_s), 3, color, 3)
        cv.circle(img3, (x_d+w1, y_d), 3, color, 3)
        cv.line(img3, (x_s, y_s), (x_d+w1, y_d), color, 3)
    cv.imshow(' ', img3)
    cv.imwrite('dennabildenefin.png', img1)
    cv.waitKey(0)

def pad(img, dh, dw):
    return cv.copyMakeBorder(img, 0, dh, 0, dw, cv.BORDER_REPLICATE )



H, _ = cv.findHomography(pts_src, pts_dst, cv.RANSAC, 7)
H = np.array(H)


plane = cv.imread('Multi-class_Yolov5_DeepSort_Pytorch-master/Mapping/plane.png')
cap = cv.VideoCapture('Multi-class_Yolov5_DeepSort_Pytorch-master/videos/OneAGV/VideoTL.mkv')

for _ in range(200):
    ret, frame = cap.read()
h, w, _ = frame.shape
warped = cv.warpPerspective(frame, H, (w,h))
drawMatches(frame, plane, pts_src, pts_dst)
#warped = frame

"""for pts_s, pts_d in zip(pts_src, pts_dst):
    xs, ys = pts_s
    xd, yd = pts_d

    color = (np.random.rand()*255, np.random.rand()*255, np.random.rand()*255)

    cv.circle(frame, (xs, ys), 3, color,3)"""

cv.imshow('',warped)
cv.waitKey(0)

