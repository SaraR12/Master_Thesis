import matplotlib.pyplot as plt
import numpy as np
import csv

xscale = 50 / 1788
yscale = 30 / 1069

xU = [371.0, 370.61022232343646, 370.4883259283365,  370.8256708333796]

def pxTom(x,y):
    print('x (m) = ', x * xscale, 'y (m) = ', (1069 - y) * yscale)

xU, yU, xP, yP = [], [], [], []

with open('update.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        xU.append(float(row[0]))
        yU.append(float(row[1]))
with open('predict.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        xP.append(float(row[0]))
        yP.append(float(row[1]))

plt.plot(np.array(xP), np.array(yP))
plt.plot(np.array(xU), np.array(yU), 'r')

plt.show()