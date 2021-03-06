from numpy import array# array([[],[],[],[],[],[],[]])

''' Homographies for the different cameras '''

def getKeypoints(camera):
    if camera == 'ME':
        pts_src = array([[747,697],[1105,696],[750,415],[1080,427],[1053,132],[1680,435],[144,408]]) # Camera
        pts_dst = array([[996,404],[998,564],[1132,400],[1127,565],[1305,565],[1128,859],[1132,104]]) # Overview
        return pts_src, pts_dst

    elif camera == 'NS':
        pts_src = array([[710,519],[720,346],[1164,574],[967,96],[1389,106],[735,86],[281,394]])
        pts_dst = array([[1609,564],[1491,564],[1638,260],[1309,400],[1309,104],[1305,564],[1532,860]])
        return pts_src, pts_dst

    elif camera == 'M':
        pts_src = array([[557,738],[1548,405],[1037,113],[1321,14],[618,57],[425,521],[524,278]])
        pts_dst = array([[146,488],[332,966],[608,709],[748,941],[683,399],[257,396],[433,396]])
        return pts_src, pts_dst

    elif camera == 'WN':
        pts_src = array([[535, 150], [720, 823], [1215, 385], [1561, 756], [1495, 209], [1835, 609], [1434, 56]])
        pts_dst = array([[149, 484], [335, 963], [612, 706], [752, 937], [821, 565], [919, 859], [821, 404]])
        return pts_src, pts_dst

    elif camera == 'WN2':
        pts_src = array([[481,396],[289,530],[1129,935],[1825,386],[1510,674],[1042,372],[1835,674]])
        pts_dst = array([[257,395],[149,485],[612,706],[997,403],[821,564],[564,397],[997,564]])
        return pts_src, pts_dst

    elif camera == 'MW':
        pts_src = array([[1455,240],[126,276],[865,557],[859,264],[1705,554],[869,964],[1107,557]])
        pts_dst = array([[1309,400],[564,397],[997,564],[997,403],[1264,564],[997,786],[1127,564]])
        #pts_dst = array([[2562,783], [1104,780], [1956, 1105], [1951,792], [2802, 1104], [1955,1539], [2206,1104]])
        return pts_src, pts_dst

    elif camera == 'EN':
        pts_src = array([[1500,886],[746,292],[676,784],[1260,692],[726,414],[133,871],[1174,19]])
        pts_dst = array([[1676,967],[1309,400],[1304,860],[1609,860],[1305,563],[998,860],[1787,0]])
        return pts_src, pts_dst

    elif camera == 'MSWOld':
        pts_src = array([[809,188],[915,663],[1635,701],[890,986],[1090,672],[1268,682],[1449,691]])
        pts_dst = array([[755,934],[998,860],[998,564],[1128,860],[998,786],[998,712],[998,638]])
        return pts_src, pts_dst

    elif camera == 'C':
        pts_src = array([[771, 440], [758, 141], [1435, 216], [666, 960], [895, 14], [1486, 216], [1333,884]])
        pts_dst = array([[611, 706], [335, 963], [149, 484], [821, 564], [0, 1068], [683, 398], [996, 404]])
        return pts_src, pts_dst
    elif camera == '1':
        pts_src = array([[316,712],[983,453],[1102,606],[161,249],[1654,250],[1101,854],[1513,607]])
        pts_dst = array([[332,960],[146,481],[256,396],[0,1069],[0,0],[433,397],[256,100]])
        return pts_src, pts_dst
    elif camera == '2':
        pts_src = array([[760,906],[1082,711],[723,325],[1390,67],[861,998],[1911,995],[1508,211]])
        pts_dst = array([[755,934],[615,703],[338,960],[153,481],[821,860],[820,108],[256,397]])
        return pts_src, pts_dst
    elif camera == '3':
        pts_src = array([[639,334],[1072,189],[841,631],[845,1058],[1065,628],[317,538],[1478,627]])
        pts_dst = array([[609,709],[506,398],[821,564],[1128,564],[820,403],[755,941],[819,108]])
        return pts_src, pts_dst
    elif camera == 'MSW':
        pts_src = array([[405,563],[1695,600],[504,461],[752,48],[1358,47],[933,461],[1852,755]])
        pts_dst = array([[749,934],[1676,960],[822,860],[999,567],[1433,566],[1129,858],[1787,1068]])
        return pts_src, pts_dst
    ######################################### EVALUATION ###################################################
    elif camera == 'BL':
        pts_src = array([[821, 646],[1304,632],[1659,628],[822,194],[641,832],[1312,201],[1719,443]])
        pts_dst = array([[153,921],[563,909],[865,906],[153,537],[0,1080],[570,543],[915,749]])
        return pts_src, pts_dst

    elif camera == 'ML':
        pts_src = array([[798,940],[1290,737],[796,22],[1291,31],[1290,495],[1635,913],[807,479]])
        pts_dst = array([[153,921],[572,749],[151,141],[572,149],[572,544],[864,898],[160,530]])
        return pts_src, pts_dst

    elif camera == 'ML2':
        #pts_src = array([[132,655],[1668,657],[344,442],[888,440],[1426,441],[1435,431],[355,431]])
        #pts_src = array([[148,629], [1655,629], [346,417], [888,415],[1424,415], [1434,405], [356,407]])
        pts_src = array([[378,629], [1887,629], [576,417],[1119,415], [1656,415], [1666,405], [586,407]])
        pts_dst = array([[0,0],     [0,1080],   [151,141],[153,530],  [153,914],  [160,921],[158,148]])
        return pts_src, pts_dst

    elif camera == 'MM':
        pts_src = array([[469,341],[964,341],[471,798],[283,166],[1553,166],[954,805],[1361,805]])
        pts_dst = array([[158,148],[579,148],[160,537],[0,0],[1080,0],[571,542],[915,543]])
        return pts_src, pts_dst

    elif camera == 'MR':
        pts_src = array([[164,160],[660,169],[176,617],[166,1078],[1064,633],[657,634],[860,633]])
        pts_dst = array([[151,141],[572,149],[160,530],[152,921],[914,543],[572,543],[743,543]])
        return pts_src, pts_dst

    elif camera == 'TL3':
        pts_src = array([[791,411],[1286,411],[793,868],[613,236],[1883,236],[1285,876],[1691,875]])
        pts_dst = array([[151,148],[572,149],[153,537],[0,0],[1080,0],[571,543],[915,543]])
        return pts_src, pts_dst

    elif camera == 'TL':
        pts_src = array([[1723,737],[67,737],[1468,522],[850,522],[864,2],[412,3],[639,3]])
        pts_dst = array([[0,0],[1080,0],[151,141], [572,141],[572,542],[914,542],[743,542]])
        return pts_src, pts_dst

    elif camera == 'TL2':
        #pts_src = array([[6,589],[1317,589],[1133,417],[1125,409],[622,417],[614,409],[1125,417]])
        pts_src = array([[96,660],[1699,660],[1474,450],[1462,438],[849,449],[838,438],[1463,449]])
        pts_dst = array([[1080,0],[0,0],     [151,141], [158,148], [572,141],[579,148],[159,141]])
        return pts_src, pts_dst

    elif camera == 'TR':
        pts_src = array([[167,477],[671,477],[169,944],[661,951],[1066,951],[1259,311],[863,952]])
        pts_dst = array([[151,141],[580,141],[153,538],[572,543],[915,543],[1079,0],[743,542]])
        return pts_src, pts_dst

    elif camera == 'CameraShelf':
        #pts_src = array([[704,957],[701,602],[687,119],[1139,120],[1596,118],[1596,613],[1132,603]])
        pts_src = array([[666,1019],[662,611],[646,57],[1165,58],[1690,56],[1690,624],[1157,612]])
        pts_dst = array([[872,906],[570,909],[160,921],[160,537],[158,149],[579,148],[571,543]])
        return pts_src, pts_dst