import numpy as np

def getKeypoints(camera):
    if camera == 'CameraWest':
        pts_src = np.array([[505, 723], [1164, 411], [1248, 208], [810, 285], [76, 711], [687, 461], [878, 410]])  # Camera
        pts_dst = np.array([[327, 401], [892, 864], [1857, 1075], [1378, 404], [327, 105], [753, 402], [891, 568]])  # Overview
        return pts_src, pts_dst
    elif camera == 'CameraMiddelFacingWestSouth':
        pts_src = np.array([[1544, 318], [1332,884], [666,958], [895,14], [1419,649], [1575,1016], [121,595]]) # Camera
        pts_dst = np.array([[327,401], [753,402], [891,568], [70,1076], [635,402], [753,329], [912,864]]) # Overviwe
        return pts_src, pts_dst