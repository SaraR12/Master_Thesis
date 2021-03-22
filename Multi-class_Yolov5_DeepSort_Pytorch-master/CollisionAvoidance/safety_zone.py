import math

def getSafetyZone(centerList, headingList, class_list, heatmap):
    points_list = []
    for center, heading, cls in zip(centerList, headingList, class_list):
        # Different size of bounding box depending on class

        xheading = heading[0]  # *0.05*safetyFactor
        yheading = heading[1]  # *0.05*safetyFactor

        threshold = 15
        threshold1 = 2

        if cls == 0 and abs(abs(xheading) - abs(yheading)) < threshold1:
            w = 60
            h = 60
        elif cls == 0 and abs(yheading) < abs(xheading):  # AGV
            w = 60
            h = 40
        elif cls == 0 and abs(xheading) < abs(yheading):
            w = 40
            h = 60
        elif cls == 1:  # Human
            w = 35
            h = 35
        x = center[0]
        y = center[1]


        #rect = Rectangle(x, y, w, h, angle)

        p0 = [x - w/2, y - h/2]  # TL
        p1 = [x + w/2, y - h/2]  # TR
        p2 = [x + w/2, y + h/2]  # BR
        p3 = [x - w/2, y + h/2]  # BL

        stopBox_color = (0,0,255)

        points_list.append([p0.copy(), p1.copy(), p2.copy(), p3.copy(), p0.copy(), [center[0], center[1]], stopBox_color])

        # Difference in x and y
        #xheading = math.ceil(heading[0])
        #yheading = math.ceil(heading[1])

        safetyFactor = heatmap[round(center[1]), round(center[0])]


        #threshold2 = 1

        if any(heading > 0) and cls == 0:

            if abs(abs(xheading) - abs(yheading)) < threshold and all([xheading, yheading]):
                print('threshold', threshold)
                if xheading > 0 and yheading > 0:
                    p1[0] += round(xheading*4)
                    p2[0] += round(xheading*4)
                    p2[1] += round(yheading*4)
                    p3[1] += round(yheading*4)
                elif xheading > 0 and yheading < 0:
                    p0[1] += round(yheading*4)
                    p1[1] += round(yheading*4)
                    p1[0] += round(xheading*4)
                    p2[0] += round(xheading*4)
                elif xheading < 0 and yheading < 0:
                    p0[0] += round(xheading*4)
                    p0[1] += round(yheading*4)
                    p1[1] += round(yheading*4)
                    p3[0] += round(xheading*4)
                elif xheading < 0 and yheading > 0:
                    p0[0] += round(xheading*4)
                    p3[0] += round(xheading*4)
                    p3[1] += round(yheading*4)
                    p2[1] += round(yheading*4)

            elif abs(xheading) > abs(yheading):
                if xheading > 0:
                    p1[0] += round(xheading*4)
                    p2[0] += round(xheading*4)
                else:
                    p0[0] += round(xheading*4)
                    p3[0] += round(xheading*4)

            elif abs(yheading) > abs(xheading):
                if yheading > 0:
                    p2[1] += round(yheading*4)
                    p3[1] += round(yheading*4)
                else:
                    p0[1] += round(yheading*4)
                    p1[1] += round(yheading*4)


        #   p0, p1, p2, p3 = rect.rotate_rectangle(p0,p1,p2,p3,angle)
        slowDown_color = (0,128,255)
        points_list.append([p0,p1,p2,p3,p0, [center[0], center[1]], slowDown_color])
    return points_list