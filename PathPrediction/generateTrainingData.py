import numpy as np
import pickle
import cv2


import math
import matplotlib.pyplot as plt

def dataGenerator(N):

    train = []
    train_truth = []

    val = []
    val_truth = []

    for sample in range(N):
        # Set frame size:
        h,w = int(round(1069/4)), int(1788/4)

        frame = np.zeros([16,h,w], dtype=int)

        nr_of_obstacles = int(np.random.randint(low=3, high=13, size=1, dtype=int))
        for _ in range(nr_of_obstacles):
            obstacle_w, obstacle_h = int(np.random.randint(low=12, high=75, size=1)), \
                                     int(np.random.randint(low=12, high=75, size=1))
            obstacle_position_x, obstacle_position_y = int(np.random.randint(low=0, high=(int(1788/4) - obstacle_w), size=1)), \
                                                      int(np.random.randint(low=0, high=(int(float(1069/4)) - obstacle_h), size=1))

            frame[:, obstacle_position_y:obstacle_position_y + obstacle_h, obstacle_position_x:obstacle_position_x + obstacle_w] \
                = np.ones([16, obstacle_h, obstacle_w])*255

        number_of_tracks = int(np.random.randint(low=10, high=20, size=1, dtype=int))
        #print(number_of_tracks)


        for track_id in range(1, number_of_tracks):
            timestep = 0
            start_point = [int(np.random.randint(low=0, high=w, size=1, dtype=int)),
                           int(np.random.randint(low=0, high=h, size=1, dtype=int))]

            frame[timestep, start_point[1], start_point[0]] = track_id

            track_length = 16

            old_x = start_point[0]
            old_y = start_point[1]

            for i in range(0,track_length,8):
                angle = np.random.rand() * 2 * math.pi
                #print('Angle = ', angle)

                steps = 0
                while steps < 8:
                    next_x = round(old_x + 5*np.random.rand() + math.cos(angle))
                    next_y = round(old_y + 5*np.random.rand() + math.sin(angle))
                    if next_x <= 1 or next_y <= 1:
                        steps = 15000

                    if next_x + 1 < frame.shape[2] and next_x > 0 and next_y > 0 and next_y + 1 < frame.shape[1]:
                    #print(frame[next_x,next_y])
                    #print(track_id)
                        if frame[0, next_y, next_x] != 255:


                            frame[timestep, next_y, next_x] = track_id  # track_id

                            old_x, old_y = next_x, next_y
                            steps += 1
                            timestep += 1
                        else:
                            #print(angle)
                            angle += 4 * math.pi/4
                            if angle > 2 * math.pi:
                                try:
                                    frame[timestep:(i + 8), old_x, old_y] = np.ones([1, 1, 8 - steps]) * track_id
                                    timestep = i + 8
                                except:
                                    pass
                                steps = 15000
                    else:
                        #print(angle)
                        angle += math.pi  # 0.174532925
                        if angle > 2 * math.pi:
                            try:
                                frame[timestep:(i + 8), old_x, old_y]  = np.ones([1, 1, 8 - steps]) * track_id
                                timestep = i + 8
                            except:
                                pass
                            steps = 15000
                #print(track_id)
        print('N = ', sample)
        if sample < round(0.7 * N):
            train.append(frame[:8,:,:])
            train_truth.append(frame[9:,:,:])
        else:
            val.append(frame[:8,:,:])
            val_truth.append(frame[9:,:,:])
    return train, train_truth, val, val_truth
if __name__ == '__main__':
    #N = input('Enter training sequences (int): ')
    train, train_truth = dataGenerator(int(1000))

    print('h')