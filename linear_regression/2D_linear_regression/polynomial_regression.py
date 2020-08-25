import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)

# print(plt.rcParams.keys())
# load the data
x, y = [], []

# data_poly.csv has only two columns x and y; we will anyway add x0 to make two data columns with one dimension having
# a polynomial trend
with open("../data_poly.csv") as f:
    for line in f.readlines():
        i, j = line.split(',')
        x.append([1, float(i), float(i)**2])
        y.append(float(j))

# convert to arrays
x = np.array(x)
y = np.array(y)

dims = x.shape[1]
# print(dims)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.scatter(x[:, 1], x[:, 0], y)

# plt.scatter(x[:,1], y)
# plt.show()

# calculate weights
w = np.reshape(np.linalg.solve(np.matmul(x.T, x), np.matmul(x.T, y)), (dims, 1))
y_hat = np.squeeze(np.matmul(x, w))

print(y_hat.shape)
# # plot all
# plt.scatter(x[:,1], y)
# plt.plot(sorted(x[:,1]), sorted(y_hat))

ax = fig.add_subplot(121, projection="3d")
ax.scatter(x[:, 1], x[:, 0], y)
ax.plot(sorted(x[:, 1]), sorted(y_hat))
plt.show()

# model performance using r squared
d1 = y - y_hat
d2 = y - y.mean()
squared_sum_ratio = d1.dot(d1) / d2.dot(d2)

r_squared = 1 - squared_sum_ratio

# print(r_squared)
print("R^2 score: {}".format(r_squared))
# print(np.matmul(np.array([1, 83.624523]), w))
# print(np.matmul(np.array([1, 83.46224523]), w))
# print(np.matmul(np.array([1, 83.1624523]), w))
# print(np.matmul(np.array([1, 83.0072624523]), w))
# print(y)
# print(y_hat)