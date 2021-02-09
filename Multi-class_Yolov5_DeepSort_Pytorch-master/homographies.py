import numpy as np

def getKeypoints(camera):
    if camera == 'CameraWest':
        pts_src = np.array([[505, 723], [1164, 411], [1248, 208], [810, 285], [76, 711], [687, 461], [878, 410]])  # Camera
        pts_dst = np.array([[327, 401], [892, 864], [1857, 1075], [1378, 404], [327, 105], [753, 402], [891, 568]])  # Overview
        return pts_src, pts_dst