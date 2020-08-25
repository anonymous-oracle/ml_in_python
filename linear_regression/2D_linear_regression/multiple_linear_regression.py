import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load the data
x, y = [], []
with open("../data_2d.csv") as f:
    for line in f.readlines():
        d1, d2, t = line.split(',')
        # adding bias term where x0 = 1 always
        x.append([1, float(d1), float(d2)])
        y.append(float(t))

x = np.array(x)
y = np.array(y)

# data plot
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x[:,1], x[:,2], y)
# plt.show()

# calculate weights
w = np.reshape(np.linalg.solve(np.matmul(x.T, x), np.matmul(x.T, y)), (3, 1))

y_pred = np.squeeze(np.matmul(x, w))
fig = plt.figure()
ax = fig.add_subplot(122, projection='3d')
ax.scatter(x[:,1], x[:,2], y_pred)
# plt.show()

# model performance using r squared

d1 = y - y_pred
d2 = y - y.mean()
squared_sum_ratio = d1.dot(d1) / d2.dot(d2)

r_squared = 1 - squared_sum_ratio

# print(r_squared)
print("R^2 score: {}".format(r_squared))

