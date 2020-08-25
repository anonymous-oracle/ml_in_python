import numpy as np
import matplotlib.pyplot as plt

# load the data
x = []
y = []
with open("../data_1d.csv", "rt") as f:
    for line in f.readlines():
        i, j = line.strip().split(',')
        x.append(float(i))
        y.append(float(j))

# turn x and y into numpy arrays
x = np.array(x)
y = np.array(y)

# plot to view
# plt.plot(x, y, 'o')
# plt.show()

# apply the equations learnt to draw a line

# # EQUATION v1
# denominator = np.square(x).mean() - np.square(x.mean())
#
# a = (np.mean(np.multiply(x, y)) - (np.mean(x) * np.mean(y))) / denominator
# b = ((np.mean(y) * np.mean(np.square(x))) - (np.mean(x) * np.mean(np.multiply(x, y)))) / denominator
# print(a)
# print(b)
# print()

# EQUATION v2
denominator = x.dot(x) - (x.mean() * x.sum())

a = (x.dot(y) - (y.mean() * x.sum())) / denominator
b = ((y.mean() * np.square(x).sum()) - (x.mean() * x.dot(y))) / denominator
# print(a)
# print(b)

# calculated best fit y
y_pred = a * x + b

plt.plot(x, y, '.')
plt.plot(x, y_pred, 'x')

# # type 1
# squared_sum_ratio = np.square(y - y_pred).sum() / np.square(y - y.mean()).sum()

# type 2
d1 = y - y_pred
d2 = y - y.mean()
squared_sum_ratio = d1.dot(d1) / d2.dot(d2)


r_square = 1 - squared_sum_ratio
plt.legend(['R square = {:.3f}'.format(r_square)])
plt.show()