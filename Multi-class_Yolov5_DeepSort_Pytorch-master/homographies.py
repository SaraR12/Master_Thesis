import numpy as np
#np.array([[],[],[],[],[],[],[]])
def getKeypoints(camera):
    if camera == 'CameraWest':
        pts_src = np.array([[504,724], [1377,678], [878,410], [1164,411], [727,410], [1009,498], [628,547]])  # Camera
        pts_dst = np.array([[257,396], [333,966], [822,565], [822,860], [820,404], [609,709], [507,398]])  # Overview
        return pts_src, pts_dst
    elif camera == 'CameraMiddelFacingWestSouth':
        pts_src = np.array([[1333,883], [895,15], [1435,216], [771,439], [616,1017], [120,595], [1488,476]]) # Camera
        pts_dst = np.array([[683,398], [0,1068], [149,484], [612,706], [842,564], [842,860], [433,396]]) # Overviwe
        return pts_src, pts_dst
    elif camera == 'NS':
        pts_src = np.array([[929,566],[1454,821],[1320,410],[971,273],[1149,274],[930,64],[1768,277]])
        pts_dst = np.array([[1631,260],[1308,104],[1308,400],[1609,564],[1432,564],[1676,960],[822,564]])
        return pts_src, pts_dst
    elif camera == 'ME':
        pts_src = np.array([[642,126],[756,578],[1048,588],[1110,923],[742,924],[223,572],[987,259]])
        pts_dst = np.array([[1631,267],[1132,400],[1128,564],[958,564],[956,403],[1132,104],[1432,564]])
        return pts_src, pts_dst
    elif camera == 'WN':
        pts_src = np.array([[641,127],[896,815],[1322,113],[1685,643],[1314,481],[1625,767],[1625,146]])
        pts_dst = np.array([[150,485],[335,964],[683,399],[822,861],[612,707],[752,937],[956,404]])
        return pts_src, pts_dst
    elif camera == 'MW':
        pts_src = np.array([[1117,129],[557,163],[1143,664],[696,462],[248,564],[705,1070],[1696,794]])
        pts_dst = np.array([[150,485],[335,964],[683,399],[612,707],[752,937],[920,564],[683,102]])
        return pts_src, pts_dst
    elif camera == 'EN':
        pts_src = np.array([[1460,578],[121,764],[948,125],[1647,755],[42,252],[811,616],[517,191]])
        pts_dst = np.array([[1609,860],[752,937],[1309,400],[1672,964],[355,396],[1227,860],[898,404]])
        return pts_src, pts_dst
    elif camera == 'MSW':
        pts_src = np.array([[804,334],[907,808],[1640,854],[1755,34],[1840,262],[1915,487],[1267,832]])
        pts_dst = np.array([[752,937],[998,861],[998,564],[506,399],[683,399],[820,404],[998,713]])
        return pts_src, pts_dst
