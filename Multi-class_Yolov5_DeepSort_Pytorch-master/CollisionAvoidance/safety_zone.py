import math

def getSafetyZone(centerList, headingList, class_list):
    points_list = []
    for center, heading, cls in zip(centerList, headingList, class_list):
        # Different size of bounding box depending on class
        if cls == 0:  # AGV
            w = 50
            h = 50
        elif cls == 1: # Human
            w = 30
            h = 30
        x = center[0]
        y = center[1]


        #rect = Rectangle(x, y, w, h, angle)

        p0 = [x - w/2, y - h/2]  # TL
        p1 = [x + w/2, y - h/2]  # TR
        p2 = [x + w/2, y + h/2]  # BR
        p3 = [x - w/2, y + h/2]  # BL

        # Difference in x and y
        xheading = math.ceil(heading[0])
        yheading = math.ceil(heading[1])
        threshold = 4

        if any(heading > 0) and cls == 0:

            if abs(abs(xheading) - abs(yheading)) < threshold and all([xheading,yheading]):
                print('threshold', threshold)
                if xheading > 0 and yheading > 0:
                    p1[0] += xheading*4
                    p2[0] += xheading*4
                    p2[1] += yheading*4
                    p3[1] += yheading*4
                elif xheading > 0 and yheading < 0:
                    p0[1] += yheading*4
                    p1[1] += yheading*4
                    p1[0] += xheading*4
                    p2[0] += xheading*4
                elif xheading < 0 and yheading < 0:
                    p0[0] += xheading*4
                    p0[1] += yheading*4
                    p1[1] += yheading*4
                    p3[0] += xheading*4
                elif xheading < 0 and yheading > 0:
                    p0[0] += xheading*4
                    p3[0] += xheading*4
                    p3[1] += yheading*4
                    p2[1] += yheading*4

            elif abs(xheading) > abs(yheading):
                if xheading > 0:
                    p1[0] += xheading*4
                    p2[0] += xheading*4
                else:
                    p0[0] += xheading*4
                    p3[0] += xheading*4

            elif abs(yheading) > abs(xheading):
                if yheading > 0:
                    p2[1] += yheading*4
                    p3[1] += yheading*4
                else:
                    p0[1] += yheading*4
                    p1[1] += yheading*4


        #   p0, p1, p2, p3 = rect.rotate_rectangle(p0,p1,p2,p3,angle)
        points_list.append([p0,p1,p2,p3, [center[0], center[1], [1]]])
    return points_list

class Rectangle:

    def __init__(self, x, y, w, h, angle):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle

    def rotate_rectangle(self, pt0, pt1, pt2, pt3, theta):

        # Point 0
        rotated_x = math.cos(theta) * (pt0[0] - self.x) - math.sin(theta) * (pt0[1] - self.y) + self.x
        rotated_y = math.sin(theta) * (pt0[0] - self.x) + math.cos(theta) * (pt0[1] - self.y) + self.y
        point_0 = [int(rotated_x), int(rotated_y)]

        # Point 1
        rotated_x = math.cos(theta) * (pt1[0] - self.x) - math.sin(theta) * (pt1[1] - self.y) + self.x
        rotated_y = math.sin(theta) * (pt1[0] - self.x) + math.cos(theta) * (pt1[1] - self.y) + self.y
        point_1 = [int(rotated_x), int(rotated_y)]

        # Point 2
        rotated_x = math.cos(theta) * (pt2[0] - self.x) - math.sin(theta) * (pt2[1] - self.y) + self.x
        rotated_y = math.sin(theta) * (pt2[0] - self.x) + math.cos(theta) * (pt2[1] - self.y) + self.y
        point_2 = [int(rotated_x), int(rotated_y)]

        # Point 3
        rotated_x = math.cos(theta) * (pt3[0] - self.x) - math.sin(theta) * (pt3[1] - self.y) + self.x
        rotated_y = math.sin(theta) * (pt3[0] - self.x) + math.cos(theta) * (pt3[1] - self.y) + self.y
        point_3 = [int(rotated_x), int(rotated_y)]

        return point_0, point_1, point_2, point_3