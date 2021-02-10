import numpy as np

def getKeypoints(camera):
    if camera == 'CameraWest':
        pts_src = np.array([[504,724], [1377,678], [878,410], [1164,411], [727,410], [1009,498], [628,547]])  # Camera
        pts_dst = np.array([[257,396], [333,966], [822,565], [822,860], [820,404], [609,709], [507,398]])  # Overview
        return pts_src, pts_dst
    elif camera == 'CameraMiddelFacingWestSouth':
        pts_src = np.array([[1333,883], [895,15], [1435,216], [771,439], [616,1017], [120,595], [1488,476]]) # Camera
        pts_dst = np.array([[683,398], [0,1068], [149,484], [612,706], [842,564], [842,860], [433,396]]) # Overviwe
        return pts_src, pts_dst