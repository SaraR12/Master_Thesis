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

    AGV2meter = np.array([[2.4747,9.682], [2.4932, 9.682], [2.5236, 9.682], [2.5654, 9.682], [2.6184, 9.682],
                          [2.6822, 9.682], [2.7564, 9.682], [2.8406, 9.682], [2.9345, 9.682], [3.0377, 9.682],
                          [3.1499, 9.682], [3.2706, 9.682], [3.3996, 9.682], [3.5364, 9.682], [3.6807, 9.682],
                          [3.8322, 9.682], [3.9004, 9.682], [4.155, 9.682], [4.3256, 9.682], [4.5018, 9.682],
                          [4.6834, 9.682], [4.8699, 9.682], [5.061, 9.682], [5.2563, 9.682], [5.4554, 9.682],
                          [5.6579, 9.682], [5.8636, 9.682], [6.072, 9.682], [6.2828, 9.682], [6.4956, 9.682],
                          [6.7101, 9.682], [6.9258, 9.682], [7.1424, 9.682], [7.3596, 9.682], [7.577, 9.682],
                          [7.7942, 9.682], [8.0108, 9.682], [8.2265, 9.682], [8.441, 9.682], [8.6538, 9.682],
                          [8.8646, 9.682], [9.073, 9.682], [9.2787, 9.682], [9.4812, 9.682],[9.6803, 9.682],
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
                          [14.932, 17.682], [15.199, 17.682], [15.475, 17.682], [15.762, 17.682],[16.056, 17.682],
                          [16.359, 17.682], [16.67, 17.682], [16.987, 17.682], [17.311, 17.682], [17.641, 17.682],
                          [17.976, 17.682], [18.316, 17.682], [18.659, 17.682], [19.007, 17.682], [19.357, 17.682],
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
                          [29.08, 17.637], [29.08, 17.581], [29.08, 17.504], [29.08, 17.407], [29.08, 17,29],
                          [29.08, 17.155], [29.08, 17.002], [29.08, 16.832], [29.08, 16.645], [29.08, 16.443],
                          [29.08, 16.226], [29.08, 15.995], [29.08, 15.751], [29.08, 15.495], [29.08, 15.226],
                          [29.08, 14.947], [29.08, 14.658], [29.08, 14.36], [29.08, 14.053], [29.08, 13.738],
                          [29.08, 13.416], [29.08, 13.088], [29.08, 12.754], [29.08, 12.416], [29.08, 12.074],
                          [29.08, 11.729], [29.08, 11.381], [29.08, 11.032], [29.08, 10.682], [29.08, 10.332],
                          [29.08, 9.9831], [29.08, 9.6356], [29.08, 9.2904], [29.08, 8.9483], [29.08, 8.6101],
                          [29.08, 8.2765], [29.08, 7.9484], [29.08, 7.6266], [29.08, 7.3117], [29.08, 7.0046],
                          [29.08, 6.706], [29.08, 6.4168], [29.08, 6.1377], [29.08, 5.8695], [29.08, 5.613],
                          [29.08, 5.3689], [29.08, 5.138], [29.08, 4.9211], [29.08, 4.7191], [29.08, 4.5325],
                          [29.08, 4,3623], [29.08, 4.2092], [29.08, 4.074], [29.08, 3.9575], [29.08, 3.8604],
                          [29.08, 3.7835], [29.08, 3.7276], [29.08, 3.6935], [29.08, 3.682], [29.08, 3.682],
                          [29.08, 3.682], [29.08, 3.682], [29.08, 3.682], [29.08, 3.682], [29.08, 3.682],
                          [29.08, 3.682], [29.08, 3.682], [29.08, 3.682], [29.08, 3.682], [29.037, 3.682],
                          [28.913, 3.682], [28.716, 3.682], [28.457, 3.682], [28.143, 3.682], [27.785, 3.682],
                          [27.39, 3.682], [26.969, 3.682], [26.529, 3.682], [26.081, 3.682], [25.632, 3.682],
                          [25.193, 3.682], [24.771, 3.682], [24.377, 3.682], [24.018, 3.682], [23.705, 3.682],
                          [23.445, 3.682], [23.249, 3.682], [23.124, 3.682], [23.081, 3.682], [23.081, 3.682]])

    AGV3Meter = np.array([[26.759, 15.701], [46.714, 15.701], [46.64, 15.701], [46.539, 15.701], [46.414, 15.701],
                 [46.265, 15.701], [46.094, 15.701], [45.903, 15.701], [45.693, 15.701], [45.466, 15.701],
                 [45.224, 15.701], [44.968, 15.701], [44.7, 15.701], [44.422, 15.701], [44.135, 15.701],
                 [43.84, 15.701], [43.54, 15.701], [43.235, 15.701], [42.928, 15.701], [42.621, 15.701],
                 [42.314, 15.701], [42.009, 15.701], [41.709, 15.701], [41.414, 15.701], [41.127, 15.701],
                 [40.849, 15.701], [40.581, 15.701], [40.325, 15.701], [40.083, 15.701], [39.856, 15.701],
                 [39.646, 15.701], [39.455, 15.701], [39.284, 15.701], [39.135, 15.701], [39.07, 15.701],
                 [38.909, 15.701], [38.835, 15.701], [38.79, 15.701], [38.775, 15.701], [38.775, 15.701],
                 [38.775, 15.701], [38.775, 15.701], [38.775, 15.701], [38.775, 15.701], [38.775, 15.701],
                 [38.775, 15.701], [38.775, 15.701], [38.775, 15.701], [38.775, 15.701], [38.775, 15.688],
                 [38.775, 15.647], [38.775, 15.582], [38.775, 15.492], [38.775, 15.378], [38.775, 15.243],
                 [38.775, 15.087], [38.775, 14.91], [38.775, 14.715], [38.775, 14.502], [38.775, 14.273],
                 [38.775, 14.028], [38.775, 13.768], [38.775, 13.495], [38.775, 13.211], [38.775, 12.915],
                 [38.775, 12.609], [38.775, 12.294], [38.775, 11.971], [38.775, 11.642], [38.775, 11.308],
                 [38.775, 10.968], [38.775, 10.626], [38.775, 10.281], [38.775, 9.9356], [38.775, 9.5899],
                 [38.775, 9.2452], [38.775, 8.9028], [38.775, 8.5637], [38.775, 8.229], [38.775, 7.8999],
                 [38.775, 7.5774], [38.775, 7.2626], [38.775, 6.9567], [38.775, 6.6608], [38.775, 6.3759],
                 [38.775, 6.1032], [38.775, 5.8438], [38.775, 5.5988], [38.775, 5.3693], [38.775, 5.1563],
                 [38.775, 4.9611], [38.775, 4.7848], [38.775, 4.6283], [38.775, 4.4929], [38.775, 4.3796],
                 [38.775, 4.2896], [38.775, 4.2239], [38.775, 4.1837], [38.775, 4.17], [38.775, 4.17],
                 [38.775, 4.17], [38.775, 4.17], [38.775, 4.17], [38.775, 4.17], [38.775, 4.17],
                 [38.775, 4.17], [38.775, 4.17], [38.775, 4.17], [38.775, 4.17], [38.764, 4.17],
                 [38.731, 4.17], [38.678, 4.17], [38.605, 4.17], [38.512, 4.17], [38.4, 4.17],
                 [38.271, 4.17], [38.123, 4.17], [37.958, 4.17], [37.777, 4.17], [37.581, 4.17],
                 [37.369, 4.17], [37.143, 4.17], [36.903, 4.17], [36.649, 4.17], [36.383, 4.17],
                 [36.105, 4.17], [35.816, 4.17], [35.516, 4.17], [35.206, 4.17], [34.887, 4.17],
                 [34.558, 4.17], [34.222, 4.17], [33.878, 4.17], [33.527, 4.17], [33.169, 4.17],
                 [32.806, 4.17], [32.438, 4.17], [32.066, 4.17], [31.69, 4.17], [31.311, 4.17],
                 [30.929, 4.17], [30.545, 4.17], [30.16, 4.17], [29.775, 4.17], [29.389, 4.17],
                 [29.004, 4.17], [28.62, 4.17], [28.238, 4.17], [27.859, 4.17], [27.483, 4.17],
                 [27.111, 4.17], [26.743, 4.17], [26.38, 4.17], [26.022, 4.17], [25.671, 4.17],
                 [25.327, 4.17], [24.991, 4.17], [24.663, 4.17], [24.343, 4.17], [24.033, 4.17],
                 [23.733, 4.17], [23.444, 4.17], [23.166, 4.17], [22.9, 4.17], [22.647, 4.17],
                 [22.406, 4.17], [22.18, 4.17], [21.968, 4.17], [21.772, 4.17], [21.591, 4.17],
                 [21.426, 4.17], [21.279, 4.17], [21.149, 4.17], [21.037, 4.17], [20.944, 4.17],
                 [20.871, 4.17], [20.818, 4.17], [20.785, 4.17], [20.774, 4.17], [20.774, 4.17],
                 [20.774, 4.17], [20.774, 4.17], [20.774, 4.17], [20.774, 4.17], [20.774, 4.17],
                 [20.774, 4.17], [20.774, 4.17], [20.774, 4.17], [20.774, 4.17], [20.763, 4.182],
                 [20.727, 4.217], [20.671, 4.2738], [20.594, 4.351], [20.497, 4.4472], [20.383, 4.5613],
                 [20.253, 4.6918], [20.107, 4.8375], [19.948, 4.997], [19.776, 5.1689], [19.592, 5.3521],
                 [19.399, 5.5451], [19.198, 5.7466], [18.989, 5.9554], [18.774, 6.17], [18.555, 6.3893],
                 [18.332, 6.6126], [18.105, 6.8392], [17.876, 7.0686], [17.644, 7.3], [17.412, 7.533],
                 [17.178, 7.7668], [16.944, 8.0009], [16.71, 8.2346], [16.477, 8.4674], [16.246, 8.6986],
                 [16.017, 8.9275], [15.791, 9.1537], [15.568, 9.3764], [15.349, 9.5951], [15.135, 9.8091],
                 [14.927, 10.018], [14.724, 10.221], [14.528, 10.417], [14.338, 10.606], [14.157, 10.788],
                 [13.984, 10.961], [13.82, 11.125], [13.665, 11.279], [13.521, 11.424], [13.387, 11.557],
                 [13.265, 11.679], [13.155, 11.789], [13.058, 11.887], [12.974, 11.971], [12.904, 12.041],
                 [12.848, 12.096], [12.808, 12.137], [12.783, 12.162], [12.774, 12.17], [12.774, 12.17],
                 [12.774, 12.17], [12.774, 12.17], [12.774, 12.17], [12.774, 12.17], [12.774, 12.17],
                 [12.774, 12.17], [12.774, 12.17], [12.774, 12.17], [12.774, 12.17], [12.774, 12.176],
                 [12.774, 12.193], [12.774, 12.222], [12.774, 12.262], [12.774, 12.315], [12.774, 12.379],
                 [12.774, 12.455], [12.774, 12.544], [12.774, 12.644], [12.774, 12.757], [12.774, 12.883],
                 [12.774, 13.021], [12.774, 13.172], [12.774, 13.335], [12.774, 13.511], [12.774, 13.701],
                 [12.774, 13.903], [12.774, 14.119], [12.774, 14.348], [12.774, 14.59], [12.774, 14.846],
                 [12.774, 15.115], [12.774, 15.398], [12.774, 15.695], [12.774, 16.006], [12.774, 16.331],
                 [12.774, 16.67], [12.774, 17.024], [12.774, 17.391], [12.774, 17.773], [12.774, 18.17],
                 [12.774, 18.581], [12.774, 19.002], [12.774, 19.431], [12.774, 19.863], [12.774, 20.295],
                 [12.774, 20.722], [12.774, 21.141], [12.774, 21.549], [12.774, 21.941], [12.774, 22.313],
                 [12.774, 22.662], [12.774, 22.985], [12.774, 23.277], [12.774, 23.534], [12.774, 23.753],
                 [12.774, 23.93], [12.774, 24.061], [12.774, 24.142], [12.774, 24.17]])

    AGV4Meter = np.array([[46.775, 26.166], [46.775, 26.155], [46.775, 26.136], [46.775, 26.11], [46.775, 26.077],
                 [46.775, 26.037], [46.775, 25.99], [46.775, 25.936], [46.775, 25.875], [46.775, 25.808],
                 [46.775, 25.734], [46.775, 25.654], [46.775, 25.568], [46.775, 25.475], [46.775, 25.376],
                 [46.775, 25.272], [46.775, 25.161], [46.775, 25.045], [46.775, 24.923], [46.775, 24.796],
                 [46.775, 24.664], [46.775, 24.526], [46.775, 24.383], [46.775, 24.235], [46.775, 24.082],
                 [46.775, 23.924], [46.775, 23.761], [46.775, 23.594], [46.775, 23.423], [46.775, 23.247],
                 [46.775, 23.067], [46.775, 22.882], [46.775, 22.694], [46.775, 22.502], [46.775, 22.306],
                 [46.775, 22.106], [46.775, 21.902], [46.775, 21.695], [46.775, 21.485], [46.775, 21.272],
                 [46.755, 21.055], [46.775, 20.835], [46.775, 20.613], [46.755, 20.388], [46.775, 20.16],
                 [46.775, 19.929], [46.775, 19.696], [46.775, 19.46], [46.775, 19.223], [46.775, 18.983],
                 [46.775, 18.741], [46.775, 18.497], [46.775, 18.252], [46.775, 18.004], [46.775, 17.756],
                 [46.775, 17.505], [46.775, 17.254], [46.775, 17.001], [46.775, 16.747], [46.775, 16.492],
                 [46.775, 16.236], [46.775, 15.979], [46.775, 15.722], [46.755, 15.464], [46.755, 15.206],
                 [46.775, 14.947], [46.775, 14.688], [46.775, 14.429], [46.775, 14.17], [46.775, 13.911],
                 [46.775, 13.652], [46.775, 13.394], [46.775, 13.136], [46.775, 12.878], [46.775, 12.622],
                 [46.755, 12.365], [46.755, 12.11], [46.775, 11.856], [46.775, 11.602], [46.775, 11.35],
                 [46.775, 11.099], [46.775, 10.849], [46.775, 10.601], [46.775, 10.355], [46.775, 10.109],
                 [46.775, 9.8662], [46.775, 9.6248], [46.775, 9.3853], [46.775, 9.1479], [46.775, 8.9127],
                 [46.775, 8.6797], [46.775, 8.4491], [46.775, 8.2209], [46.775, 7.9954], [46.775, 7.7725],
                 [46.775, 7.5523], [46.775, 7.3351], [46.775, 7.1208], [46.775, 6.9096], [46.775, 6.7016],
                 [46.775, 6.4969], [46.775, 6.2956], [46.775, 6.0978], [46.775, 5.9036], [46.775, 5.7131],
                 [46.775, 5.5263], [46.775, 5.3435], [46.775, 5.1647], [46.775, 4.99], [46.775, 4.8196],
                 [46.775, 4.6534], [46.775, 4.4916], [46.775, 4.3344], [46.775, 4.1818], [46.775, 4.034],
                 [46.775, 3.8909], [46.775, 3.7528], [46.775, 3.6197], [46.775, 3.4918], [46.775, 3.3691],
                 [46.775, 3.2517], [46.775, 3.1398], [46.775, 3.0335], [46.775, 2.9328], [46.775, 2.8378],
                 [46.775, 2.7487], [46.775, 2.6656], [46.775, 2.5885], [46.775, 2.5176], [46.775, 2.4529],
                 [46.775, 2.3946], [46.775, 2.3428], [46.775, 2.2976], [46.775, 2.259], [46.775, 2.2273],
                 [46.775, 2.2024], [46.775, 2.1845], [46.775, 2.1736], [46.775, 2.17], [46.775, 2.17],
                 [46.775, 2.17], [46.775, 2.17], [46.775, 2.17], [46.775, 2.17], [46.775, 2.17],
                 [46.775, 2.17], [46.775, 2.17], [46.775, 2.17], [46.775, 2.17], [46.775, 2.17],
                 [46.775, 2.17], [46.775, 2.17], [46.775, 2.17], [46.775, 2.17], [46.775, 2.17],
                 [46.775, 2.17], [46.775, 2.17], [46.775, 2.17], [46.775, 2.17], [46.775, 2.1773],
                 [46.775, 2.1988], [46.775, 2.2342], [46.775, 2.2831], [46.775, 2.3449], [46.775, 2.4194],
                 [46.775, 2.506], [46.775, 2.6044], [46.755, 2.7141], [46.755, 2.8347], [46.755, 2.9658],
                 [46.775, 3.107], [46.775, 3.2579], [46.775, 3.418], [46.775, 3.5869], [46.775, 3.7642],
                 [46.775, 3.9495], [46.775, 4.1423], [46.775, 4.3423], [46.775, 4.459], [46.775, 4.762],
                 [46.775, 4.9809], [46.775, 5.2052], [46.775, 5.4346], [46.775, 5.6685], [46.775, 5.9067],
                 [46.775, 6.1487], [46.775, 6.394], [46.775, 6.6423], [46.775, 6.893], [46.775, 7.1459],
                 [46.775, 7.4005], [46.775, 7.6563], [46.775, 7.9129], [46.775, 8.17], [46.775, 8.4271],
                 [46.775, 8.6837], [46.775, 8.9395], [46.775, 9.1941], [46.775, 9.447], [46.775, 9.6977],
                 [46.775, 9.946], [46.775, 10.191], [46.775, 10.433], [46.775, 10.671], [46.775, 10.905],
                 [46.775, 11.135], [46.775, 11.359], [46.755, 11.578], [46.775, 11.791], [46.775, 11.998],
                 [46.775, 12.198], [46.775, 12.391], [46.775, 12.576], [46.775, 12.753], [46.775, 12.922],
                 [46.775, 13.082], [46.775, 13.233], [46.775, 13.374], [46.775, 13.505], [46.775, 13.626],
                 [46.775, 13.736], [46.775, 13.834], [46.775, 13.921], [46.775, 13.995], [46.775, 14.057],
                 [46.775, 14.106], [46.775, 14.141], [46.775, 14.163], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17],
                 [46.775, 14.17], [46.775, 14.17], [46.775, 14.17], [46.775, 14.17]])

    Human1 = np.array([[7.6194, 6.502],[7.6194, 6.502],[7.6194, 6.502],[7.6194, 6.502],[7.6194, 6.502],
                       [7.6194, 6.502], [7.6194, 6.502], [7.6194, 6.502], [7.6194, 6.502], [7.6194, 6.502],
                       [7.6125, 6.4969], [7.5921, 6.4818], [7.5588, 6.4571], [7.5132, 64233], [7.4557, 6.3807],
                       [7.3871, 6.3298], [7.3078, 6.2711], [7.2184, 6.2048], [7.1195, 6.1315], [7.0115, 6.0515],
                       [6.8952, 5.9652], [6.771, 5.8732], [6.6395, 5.7757], [6.5013, 5.6733], [6.3569, 5.5663],
                       [6.2069, 5.4551], [6.0518, 5.3402], [5.8923, 5.2219], [5.7288, 5.1008], [5.562, 4.9771],
                       [5.3924, 4.8514], [5.2205, 4.724], [5.0469, 4.5954], [4.8722, 4.4659], [4.6969, 4.336],
                       [4.5217, 4.2061], [4.347, 4.0766], [4.1734, 3.948], [4.0015, 3.8206], [3.8319, 3.6949],
                       [3.6651, 3.5712], [3.5016, 3.4501], [3.3421, 3.3318], [3.187, 3.2169], [3.037, 3.1057],
                       [2.8926, 2.9987], [2.7544, 2.8963], [2.6229, 2.7988], [2.4987, 2.7068], [2.3824, 2.6205],
                       [2.2744, 2.5405], [2.1755, 2.4672], [2.0861, 2.4009], [2.0068, 2.3422], [1.9382, 2.2913],
                       [1.8807, 2.2487], [1.8351, 2.2149], [1.8018, 2.1902], [1.7814, 2.1751], [1.7745, 2.17],
                       [1.7745, 2.17], [1.7745, 2.17], [1.7745, 2.17], [1.7745, 2.17], [1.7745, 2.17],
                       [1.7745, 2.17], [1.7745, 2.17], [1.7745, 2.17], [1.7745, 2.17], [1.7745, 2.17],
                       [1.7784, 2.17], [1.7899, 2.17], [1.8091, 2.17], [1.8356, 2.17], [1.8694, 2.17],
                       [1.9103, 2.17], [1.9582, 2.17], [2.013, 2.17], [2.0745, 2.17], [2.1426, 2.17],
                       [2.2172, 2.17], [2.298, 2.17], [2.3851, 2.17], [2.4782, 2.17], [2.5772, 2.17],
                       [2.682, 2.17], [2.7925, 2.17], [2.9084, 2.17], [3.0297, 2.17], [3.1563, 2.17],
                       [3.2879, 2.17], [3.4245, 2.17], [3.566, 2.17], [3.7121, 2.17], [3.8627, 2.17],
                       [4.0178, 2.17], [4.1772, 2.17], [4.3407, 2.17], [4.5082, 2.17], [4.6796, 2.17],
                       [4.8548, 2.17], [5.0335, 2.17], [5.2157, 2.17], [5.4012, 2.17], [5.59, 2.17],
                       [5.7817, 2.17], [5.9765, 2.17], [6.1739, 2.17], [6.3741, 2.17], [6.5767, 2.17],
                       [6.7818, 2.17], [6.9891, 2.17], [7.1984, 2.17], [7.4098, 2.17], [7.623, 2.17],
                       [7.8379, 2.17], [8.0543, 2.17], [8.2722, 2.17], [8.4914, 2.17], [8.7117, 2.17],
                       [8.933, 2.17], [9.1553, 2.17], [9.3783, 2.17], [9.6018, 2.17], [9.8259, 2.17],
                       [10.05, 2.17], [10.275, 2.17], [10.5, 2.17], [10.724, 2.17], [10.949, 2.17],
                       [11.173, 2.17], [11.396, 2.17], [11.619, 2.17], [11.841, 2.17], [12.063, 2.17],
                       [12.283, 2.17], [12.502, 2.17], [12.72, 2.17], [12.937, 2.17], [13.152, 2.17],
                       [13.365, 2.17], [13.576, 2.17], [13.785, 2.17], [13.993, 2.17], [14.198, 2.17],
                       [14.4, 2.17], [14.601, 2.17], [14.798, 2.17], [14.993, 2.17], [15.185, 2.17],
                       [15.373, 2.17], [15.559, 2.17], [15.741, 2.17], [15.92, 2.17], [16.095, 2.17],
                       [16.266, 2.17], [16.434, 2.17], [16.597, 2.17], [16.757, 2.17], [16.912, 2.17],
                       [17.062, 2.17], [17.209, 2.17], [17.35, 2.17], [17.487, 2.17], [17.618, 2.17],
                       [17.745, 2.17], [17.866, 2.17], [17.982, 2.17], [18.092, 2.17], [18.197, 2.17],
                       [18.296, 2.17], [18.389, 2.17], [18.476, 2.17], [18.557, 2.17], [18.632, 2.17],
                       [18.7, 2.17], [18.762, 2.17], [18.816, 2.17], [18.864, 2.17], [18.905, 2.17],
                       [18.939, 2.17], [18.965, 2.17], [18.985, 2.17], [18.996, 2.17], [19, 2.17],
                       [19, 2.17], [19, 2.17], [19, 2.17], [19, 2.17], [19, 2.17],
                       [19, 2.17], [19, 2.17], [19, 2.17], [19, 2.17], [19, 2.17],
                       [19, 2.17], [19, 2.17], [19, 2.17], [19, 2.17], [19, 2.17],
                       [19, 2.1741], [19, 2.1864], [19, 2.2066], [19, 2.2346], [19, 2.2703],
                       [19, 2.3134], [19, 2.3638], [19, 2.4214], [19, 2.4859], [19, 2.5572],
                       [19, 2.6352], [19, 2.7197], [19, 2.8104], [19, 2.9073], [19, 3.0102],
                       [19, 3.1188], [19, 3.2332], [19, 3.353], [19, 3.4781], [19, 3.6083],
                       [19, 3.7436], [19, 3.8836], [19, 4.0283], [19, 4.1775], [19, 4.3309],
                       [19, 4.4886], [19, 4.6502], [19, 4.8156], [19, 4.9847], [19, 5.1573],
                       [19, 5.3332], [19, 5.5122], [19, 5.6942], [19, 5.8791], [19, 6.0666],
                       [19, 6.2566], [19, 6.4489], [19, 6.6434], [19, 6.8399], [19, 7.0382],
                       [19, 7.2381], [19, 7.4396], [19, 7.6423], [19, 7.8463], [19, 8.0512],
                       [19, 8.257], [19, 8.4634], [19, 8.6703], [19, 8.8776], [19, 9.085],
                       [19, 9.2924], [19, 9.4997], [19, 9.7066], [19, 9.913], [19, 10.119],
                       [19, 10.324], [19, 10.528], [19, 10.73], [19, 10.932], [19, 11.132],
                       [19, 11.33], [19, 11.527], [19, 11.721], [19, 11.913], [19, 12.103],
                       [19, 12.291], [19, 12.476], [19, 12.658], [19, 12.837], [19, 13.013],
                       [19, 13.185], [19, 13.354], [19, 13.52], [19, 13.681], [19, 13.839],
                       [19, 13.993], [19, 14.142], [19, 14.286], [19, 14.426], [19, 14.562],
                       [19, 14.692], [19, 14.817], [19, 14.937], [19, 15.051], [19, 15.16],
                       [19, 15.263], [19, 15.36], [19, 15.45], [19, 15.535], [19, 15.613],
                       [19, 15.684], [19, 15.749], [19, 15.806], [19, 15.857], [19, 15.9],
                       [19, 15.935], [19, 15.963], [19, 15.984], [19, 15.996], [19, 16]])
    Human2 = np.array([[28.774, 18.17], [28.774, 18.162], [28.774, 18.139], [28.774, 18.103], [28.774, 18.056],
                   [28.774, 17.999], [28.774, 17.934], [28.774, 17.863], [28.774, 17.787], [28.774, 17.709],
                   [28.774, 17.631], [28.774, 17.553], [28.774, 17.477], [28.774, 17.406], [28.774, 17.341],
                   [28.774, 17.284], [28.774, 17.237], [28.774, 17.201], [28.774, 17.178], [28.774, 17.17],
                   [28.774, 17.17], [28.774, 17.17], [28.774, 17.17], [28.774, 17.17], [28.774, 17.17],
                   [28.774, 17.17], [28.774, 17.17], [28.774, 17.17], [28.774, 17.17], [28.774, 17.17],
                   [28.766, 17.17], [28.741, 17.17], [28.7, 17.17], [28.644, 17.17], [28.572, 17.17],
                   [28.486, 17.17], [28.385, 17.17], [28.27, 17.17], [28.142, 17.17], [28.001, 17.17],
                   [27.847, 17.17], [27.681, 17.17], [27.503, 17.17], [27.314, 17.17], [27.114, 17.17],
                   [26.903, 17.17], [26.683, 17.17], [26.452, 17.17], [26.213, 17.17], [25.965, 17.17],
                   [25.708, 17.17], [25.443, 17.17], [25.171, 17.17], [24.892, 17.17], [24.606, 17.17],
                   [24.314, 17.17], [24.016, 17.17], [23.713, 17.17], [23.404, 17.17], [23.091, 17.17],
                   [22.774, 17.17], [22.454, 17.17], [22.131, 17.17], [21.806, 17.17], [21.481, 17.17],
                   [21.157, 17.17], [20.834, 17.17], [20.514, 17.17], [20.198, 17.17], [19.887, 17.17],
                   [19.581, 17.17], [19.283, 17.17], [18.993, 17.17], [18.713, 17.17], [18.442, 17.17],
                   [18.183, 17.17], [17.937, 17.17], [17.704, 17.17], [17.485, 17.17], [17.283, 17.17],
                   [17.097, 17.17], [16.929, 17.17], [16.779, 17.17], [16.65, 17.17], [16.542, 17.17],
                   [16.456, 17.17], [16.393, 17.17], [16.355, 17.17], [16.342, 17.17], [16.342, 17.17],
                   [16.342, 17.17], [16.342, 17.17], [16.342, 17.17], [16.342, 17.17], [16.342, 17.17],
                   [16.342, 17.17], [16.342, 17.17], [16.342, 17.17], [16.342, 17.17], [16.342, 17.17],
                   [16.335, 17.167], [16.313, 17.157], [16.279, 17.142], [16.231, 17.12], [16.172, 17.093],
                   [16.1, 17.061], [16.019, 17.024], [15.926, 16.981], [15.825, 16.934], [15.714, 16.883],
                   [15.596, 16.827], [15.469, 16.767], [15.336, 16.704], [15.197, 16.636], [15.051, 16.566],
                   [14.901, 16.492], [14.747, 16.416], [14.589, 16.337], [14.427, 16.225], [14.264, 16.172],
                   [14.098, 16.086], [13.932, 15.998], [13.765, 15.909], [13.598, 15.819], [13.432, 15.728],
                   [13.268, 15.636], [13.105, 15.543], [12.946, 15.45], [12.79, 15.356], [12.638, 15.263],
                   [12.491, 15.17], [12.347, 15.077], [12.205, 14.985], [12.064, 14.894], [11.926, 14.803],
                   [11.789, 14.713], [11.653, 14.623], [11.52, 14.533], [11.387, 14.444], [11.257, 14.356],
                   [11.128, 14.268], [11.001, 14.181], [10.875, 14.094], [10.751, 14.008], [10.629, 13.922],
                   [10.508, 13.837], [10.388, 13.752], [10.271, 13.668], [10.154, 13.584], [10.04, 13.501],
                   [9.9265, 13.418], [9.8148, 13.336], [9.7047, 13.254], [9.596, 13.173], [9.4887, 13.093],
                   [9.383, 13.013], [9.2787, 12.933], [9.1759, 12.854], [9.0744, 12.776], [8.9745, 12.698],
                   [8.8759, 12.62], [8.7788, 12.543], [8.683, 12.467], [8.5886, 12.391], [8.4956, 12.316],
                   [8.404, 12.241], [8.3138, 12.167], [8.2248, 12.093], [8.1373, 12.02], [8.051, 11.947],
                   [7.9661, 11.875], [7.8824, 11.804], [7.8001, 11.733], [7.7191, 11.662], [7.6393, 11.592],
                   [7.5608, 11.523], [7.4835, 11.454], [7.4075, 11.385], [7.3328, 11.318], [7.2592, 11.25],
                   [7.1869, 11.183], [7.1158, 11.117], [7.0459, 11.052], [6.9771, 10.986], [6.9096, 10.922],
                   [6.8432, 10.858], [6.7779, 10.794], [6.7138, 10.731], [6.6509, 10.669], [6.589, 10.607],
                   [6.5283, 10.545], [6.4687, 10.484], [6.4101, 10.424], [6.3527, 10.364], [6.2963, 10.305],
                   [6.241, 10.246], [6.1868, 10.188], [6.1335, 10.13], [6.0814, 10.073], [6.0302, 10.017],
                   [5.9801, 9.9608], [5.9309, 9.9053], [5.8828, 9.8504], [5.8356, 9.796], [5.7894, 9.7422],
                   [5.7442, 9.6889], [5.6999, 9.6362], [5.6566, 9.584], [5.6142, 9.5324], [5.5727, 9.4813],
                   [5.5321, 9.4307], [5.4925, 9.3807], [5.4537, 9.3313], [5.4158, 9.2824], [5.3788, 9.234],
                   [5.3426, 9.1862], [5.3073, 9.1389], [5.2728, 9.0922], [5.2392, 9.046], [5.2064, 9.0004],
                   [5.1744, 8.9554], [5.1431, 8.9109], [5.1127, 8.8669], [5.0831, 8.8235], [5.0542, 8.7806],
                   [5.0261, 8.7383], [4.9987, 8.6966], [4.9721, 8.6554], [4.9462, 8.6148], [4.9211, 8.5747],
                   [4.8966, 8.5351], [4.8728, 8.4962], [4.8498, 8.4577], [4.8274, 8.4199], [4.8056, 8.3826],
                   [4.7846, 8.3458], [4.7641, 8.3096], [4.7444, 8.274], [4.7252, 8.2389], [4.7067, 8.2044],
                   [4.6887, 8.1704], [4.6714, 8.137], [4.6546, 8.1042], [4.6385, 8.0719], [4.6229, 8.0402],
                   [4.6078, 8.009], [4.5933, 7.9784], [4.5794, 7.9484], [4.5659, 7.9189], [4.553, 7.89],
                   [4.5406, 7.8616], [4.5287, 7.8338], [4.5173, 7.8066], [4.5063, 7.78], [4.4958, 7.7539],
                   [4.4858, 7.7283], [4.4762, 7.7034], [4.4671, 7.679], [4.4584, 7.6551], [4.45, 7.6319],
                   [4.4421, 7.6092], [4.4346, 7.587], [4.4275, 7.5654], [4.4208, 7.5444], [4.4144, 7.524],
                   [4.4084, 7.5041], [4.4027, 7.4849], [4.3973, 7.4661], [4.3923, 7.448], [4.3876, 7.4304],
                   [4.3832, 7.4134], [4.3791, 7.3969], [4.3752, 7.3811], [4.3717, 7.3658], [4.3684, 7.351],
                   [4.3653, 7.3369], [4.3625, 7.3233], [4.36, 7.3103], [4.3576, 7.2979], [4.3555, 7.286],
                   [4.3536, 7.2747], [4.3518, 7.264], [4.3503, 7.2539], [4.3489, 7.2443], [4.3477, 7.2353],
                   [4.3466, 7.2269], [4.3457, 7.2191], [4.3449, 7.2118], [4.3442, 7.2052], [4.3437, 7.1991],
                   [4.3432, 7.1935], [4.3429, 7.1886], [4.3426, 7.1843], [4.3424, 7.1805], [4.3422, 7.1773],
                   [4.3421, 7.1747], [4.342, 7.1726], [4.342, 7.1712], [4.342, 7.1703], [4.342, 7.17]])
    
    return  AGV1meter, AGV2meter, Human1

#print(len(getGroundTruth()))