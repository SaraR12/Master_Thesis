import numpy as np
# np.array([[],[],[],[],[],[],[]])

''' Homographies for the different cameras '''

def getKeypoints(camera):
    if camera == 'ME':
        pts_src = np.array([[747,697],[1105,696],[750,415],[1080,427],[1053,132],[1680,435],[144,408]]) # Camera
        pts_dst = np.array([[996,404],[998,564],[1132,400],[1127,565],[1305,565],[1128,859],[1132,104]]) # Overview
        return pts_src, pts_dst

    elif camera == 'NS':
        pts_src = np.array([[710,519],[720,346],[1164,574],[967,96],[1389,106],[735,86],[281,394]])
        pts_dst = np.array([[1609,564],[1491,564],[1638,260],[1309,400],[1309,104],[1305,564],[1532,860]])
        return pts_src, pts_dst

    elif camera == 'M':
        pts_src = np.array([[557,738],[1548,405],[1037,113],[1321,14],[618,57],[425,521],[524,278]])
        pts_dst = np.array([[146,488],[332,966],[608,709],[748,941],[683,399],[257,396],[433,396]])
        return pts_src, pts_dst

    elif camera == 'WN':
        pts_src = np.array([[535, 150], [720, 823], [1215, 385], [1561, 756], [1495, 209], [1835, 609], [1434, 56]])
        pts_dst = np.array([[149, 484], [335, 963], [612, 706], [752, 937], [821, 565], [919, 859], [821, 404]])
        return pts_src, pts_dst

    elif camera == 'WN2':
        pts_src = np.array([[481,396],[289,530],[1129,935],[1825,386],[1510,674],[1042,372],[1835,674]])
        pts_dst = np.array([[257,395],[149,485],[612,706],[997,403],[821,564],[564,397],[997,564]])
        return pts_src, pts_dst

    elif camera == 'MW':
        pts_src = np.array([[1455,240],[126,276],[865,557],[859,264],[1705,554],[869,964],[1107,557]])
        pts_dst = np.array([[1309,400],[564,397],[997,564],[997,403],[1264,564],[997,786],[1127,564]])
        #pts_dst = np.array([[2562,783], [1104,780], [1956, 1105], [1951,792], [2802, 1104], [1955,1539], [2206,1104]])
        return pts_src, pts_dst

    elif camera == 'EN':
        pts_src = np.array([[1500,886],[746,292],[676,784],[1260,692],[726,414],[133,871],[1174,19]])
        pts_dst = np.array([[1676,967],[1309,400],[1304,860],[1609,860],[1305,563],[998,860],[1787,0]])
        return pts_src, pts_dst

    elif camera == 'MSWOld':
        pts_src = np.array([[809,188],[915,663],[1635,701],[890,986],[1090,672],[1268,682],[1449,691]])
        pts_dst = np.array([[755,934],[998,860],[998,564],[1128,860],[998,786],[998,712],[998,638]])
        return pts_src, pts_dst

    elif camera == 'C':
        pts_src = np.array([[771, 440], [758, 141], [1435, 216], [666, 960], [895, 14], [1486, 216], [1333,884]])
        pts_dst = np.array([[611, 706], [335, 963], [149, 484], [821, 564], [0, 1068], [683, 398], [996, 404]])
        return pts_src, pts_dst
    elif camera == '1':
        pts_src = np.array([[316,712],[983,453],[1102,606],[161,249],[1654,250],[1101,854],[1513,607]])
        pts_dst = np.array([[332,960],[146,481],[256,396],[0,1069],[0,0],[433,397],[256,100]])
        return pts_src, pts_dst
    elif camera == '2':
        pts_src = np.array([[760,906],[1082,711],[723,325],[1390,67],[861,998],[1911,995],[1508,211]])
        pts_dst = np.array([[755,934],[615,703],[338,960],[153,481],[821,860],[820,108],[256,397]])
        return pts_src, pts_dst
    elif camera == '3':
        pts_src = np.array([[639,334],[1072,189],[841,631],[845,1058],[1065,628],[317,538],[1478,627]])
        pts_dst = np.array([[609,709],[506,398],[821,564],[1128,564],[820,403],[755,941],[819,108]])
        return pts_src, pts_dst
    elif camera == 'MSW':
        pts_src = np.array([[405,563],[1695,600],[504,461],[752,48],[1358,47],[933,461],[1852,755]])
        pts_dst = np.array([[749,934],[1676,960],[822,860],[999,567],[1433,566],[1129,858],[1787,1068]])
        return pts_src, pts_dst
