import numpy as np

class HeatMap:
    def __init__(self, img, timesteps):
        self.h, self.w, _ = img.shape
        self.timesteps = timesteps
        self.map_list = []
        self.map = np.zeros((self.h, self.w))

    def update(self, position_list, class_list):

        for position, cls in zip(position_list, class_list):

            if cls == 0: # AGV
                x1 = 20
                x2 = 30
                x3 = 35
                x4 = 45
                x5 = 50
                x6 = 55
            elif cls == 1: # Human
                x1 = 15
                x2 = 20
                x3 = 25
                x4 = 30
                x5 = 35
                x6 = 35
            if self.map[round(position[1]), round(position[0])] < 6:
                self.map[round(position[1])-x1:round(position[1])+x1,round(position[0])-x1:round(position[0])+x1] += np.ones((x1*2,x1*2))*6
                self.map[round(position[1])-x2:round(position[1])+x2,round(position[0])-x2:round(position[0])+x2] += np.ones((x2*2,x2*2))*5
                self.map[round(position[1])-x3:round(position[1])+x3,round(position[0])-x3:round(position[0])+x3] += np.ones((x3*2,x3*2))*4
                self.map[round(position[1])-x4:round(position[1])+x4,round(position[0])-x4:round(position[0])+x4] += np.ones((x4*2,x4*2))*3
                self.map[round(position[1])-x5:round(position[1])+x5,round(position[0])-x5:round(position[0])+x5] += np.ones((x5*2,x5*2))*2
                self.map[round(position[1])-x6:round(position[1])+x6,round(position[0])-x6:round(position[0])+x6] += np.ones((x6*2,x6*2))
        self.map_list.append(self.map)

        if len(self.map_list) > self.timesteps:
            self.map_list.pop(0)
        self.map = np.zeros((self.h, self.w))

        #heatmap = [np.zeros((self.h, self.w)) + map for map in self.map_list][0]
        heatmap = np.zeros((self.h, self.w))
        for hmap in self.map_list:
            heatmap += hmap

        return heatmap

    def retrieve(self):
        pass

    def setStaticZones(self):
        pass