import numpy as np

def getGroundTruth():

    AGV1meter = np.array([[2.4747, 9.682], [2.4932, 9.682], [2.5236, 9.682], [2.5654, 9.682], [2.6184, 9.682],
                      [2.6822, 9.682], [2.7564, 9.682], [2.8406, 9.682], [2.9345, 9.682], [3.0377, 9.682],
                      [3.1499, 9.682], [3.2706, 9.682], [3.3996, 9.682], [3.5364, 9.682], [3.6807, 9.682],
                      [3.8322, 9.682], [3.9904, 9.682], [4.155, 9.682], [4.3256, 9.682], [4.5018, 9.682],
                      [4.6834, 9.682], [4.8699, 9.682], [5.061, 9.682], [5.2563, 9.682], [5.4554, 9.682],
                      [5.6579, 9.682], [5.8636, 9.682], [6.072, 9.682], [6.2828, 9.682], [6.4956, 9.682],
                      [6.7101, 9.682], [6.9258, 9.682], [7.1424, 9.682], [7.3596, 9.682], [7.577, 9.682],
                      [7.7942, 9.682], [8.0108, 9.682], [8.2265, 9.682], [8.441, 9.682], [8.6538, 9.682],
                      [8.8646, 9.682], [9.073, 9.682], [9.2787, 9.682], [9.4812, 9.682], [9.6803, 9.682],
                      [9.8756, 9.682], [10.067, 9.682], [10.253, 9.682], [10.435, 9.682], [10.611, 9.682],
                      [10.782, 9.682], [10.946, 9.682], [11.104, 9.682], [11.256, 9.682], [11.4, 9.682],
                      [11.537, 9.682], [11.666, 9.682], [11.787, 9.682], [11.899, 9.682], [12.002, 9.682],
                      [12.096, 9.682], [12.18, 9.682], [12.254, 9.682], [12.318, 9.682], [12.371, 9.682],
                      [12.413, 9.682], [12.443, 9.682], [12.462, 9.682], [12.468, 9.682], [12.468, 9.682],
                      [12.468, 9.682], [12.468, 9.682], [12.468, 9.682], [12.468, 9.682], [12.468, 9.682],
                      [12.468, 9.682], [12.468, 9.682], [12.468, 9.682], [12.468, 9.682], [12.468, 9.6968],
                      [12.468, 9.74], [12.468, 9.8103], [12.468, 9.906], [12.468, 10.026], [12.468, 10.168],
                      [12.468, 10.331], [12.468, 10.514], [12.468, 10.715], [12.468, 10.932], [12.468, 11.164],
                      [12.468, 11.41], [12.468, 11.668], [12.468, 11.936], [12.468, 12.213], [12.468, 12.498],
                      [12.468, 12.789], [12.468, 13.084], [12.468, 13.382], [12.468, 13.682], [12.468, 13.982],
                      [12.468, 14.28], [12.468, 14.575], [12.468, 14.866], [12.468, 15.151], [12.468, 15.428],
                      [12.468, 15.696], [12.468, 15.954], [12.468, 16.2], [12.468, 16.432], [12.468, 16.649],
                      [12.468, 16.85], [12.468, 17.033], [12.468, 17.196], [12.468, 17.338], [12.468, 17.458],
                      [12.468, 17.554], [12.468, 17.624], [12.468, 17.667], [12.468, 17.682], [12.468, 17.682],
                      [12.468, 17.682], [12.468, 17.682], [12.468, 17.682], [12.468, 17.682], [12.468, 17.682],
                      [12.468, 17.682], [12.468, 17.682], [12.468, 17.682], [12.468, 17.682], [12.478, 17.682],
                      [12.508, 17.682], [12.557, 17.682], [12.625, 17.682], [12.71, 17.682], [12.813, 17.682],
                      [12.933, 17.682], [13.069, 17.682], [13.221, 17.682], [13.388, 17.682], [13.57, 17.682],
                      [13.765, 17.682], [13.974, 17.682], [14.196, 17.682], [14.43, 17.682], [14.675, 17.682],
                      [14.932, 17.682], [15.199, 17.682], [15.475, 17.682], [15.762, 17.682], [16.056, 17.682],
                      [16.359, 17.682], [16.67, 17.682], [16.987, 17.682], [17.311, 17.682], [17.641, 17.682],
                      [16.976, 17.682], [18.316, 17.682], [18.659, 17.682], [19.007, 17.682], [19.357, 17.682],
                      [19.709, 17.682], [20.063, 17.682], [20.418, 17.682], [20.774, 17.682], [21.13, 17.682],
                      [21.485, 17.682], [21.84, 17.682], [22.192, 17.682], [22.542, 17.682], [22.889, 17.682],
                      [23.233, 17.682], [23.573, 17.682], [23.908, 17.682], [24.237, 17.682], [24.561, 17.682],
                      [24.879, 17.682], [25.189, 17.682], [25.492, 17.682], [25.787, 17.682], [26.073, 17.682],
                      [26.35, 17.682], [26.617, 17.682], [26.874, 17.682], [27.119, 17.682], [27.353, 17.682],
                      [27.574, 17.682], [27.783, 17.682], [27.979, 17.682], [28.16, 17.682], [28.327, 17.682],
                      [28.479, 17.682], [28.615, 17.682], [28.735, 17.682], [28.838, 17.682], [28.924, 17.682],
                      [28.992, 17.682], [29.041, 17.682], [29.07, 17.682], [29.08, 17.682], [29.08, 17.682],
                      [29.08, 17.682], [29.08, 17.682], [29.08, 17.682], [29.08, 17.682], [29.08, 17.682],
                      [29.08, 17.682], [29.08, 17.682], [29.08, 17.682], [29.08, 17.682], [29.08, 17.671],
                      [29.08, 17.637], [29.08, 17.581], [29.08, 17.504], [29.08, 17.407], [29.08, 17.29],
                      [29.08, 17.155], [29.08, 17.002], [29.08, 16.832], [29.08, 16.645], [29.08, 16.443],
                      [29.08, 16.226], [29.08, 15.995], [29.08, 15.751], [29.08, 15.495], [29.08, 15.226],
                      [29.08, 14.947], [29.08, 14.658], [29.08, 14.36], [29.08, 14.053], [29.08, 13.738],
                      [29.08, 13.416], [29.08, 13.088], [29.08, 12.574], [29.08, 12.416], [29.08, 12.074],
                      [29.08, 11.729], [29.08, 11.381], [29.08, 11.032], [29.08, 10.682], [29.08, 10.332],
                      [29.08, 9.98313], [29.08, 9.6356], [29.08, 9.2904], [29.08, 8.9483], [29.08, 8.6101],
                      [29.08, 8.2765], [29.08, 7.9484], [29.08, 7.6266], [29.08, 7.3117], [29.08, 7.0046],
                      [29.08, 6.706], [29.08, 6.4168], [29.08, 6.1377], [29.08, 5.8695], [29.08, 5.613],
                      [29.08, 5.3689], [29.08, 5.138], [29.08, 4.9211], [29.08, 4.7191], [29.08, 4.5325],
                      [29.08, 4.3623], [29.08, 4.2092], [29.08, 4.074], [29.08, 3.9575], [29.08, 3.8604],
                      [29.08, 3.7835], [29.08, 3.7276], [29.08, 3.6935], [29.08, 3.682], [29.08, 3.682],
                      [29.08, 3.682], [29.08, 3.682], [29.08, 3.682], [29.08, 3.682], [29.08, 3.682],
                      [29.08, 3.682], [29.08, 3.682], [29.08, 3.682], [29.08, 3.682], [29.037, 3.682],
                      [28.913, 3.682], [28.716, 3.682], [28.457, 3.682], [28.143, 3.682], [27.785, 3.682],
                      [27.39, 3.682], [26.969, 3.682], [26.529, 3.682], [26.081, 3.682], [25.632, 3.682],
                      [25.193, 3.682], [24.771, 3.682], [24.377, 3.682], [24.018, 3.682], [23.705, 3.682],
                      [23.445, 3.682], [23.249, 3.682], [23.124, 3.682], [23.081, 3.682]])
    return AGV1meter

print(len(getGroundTruth()))