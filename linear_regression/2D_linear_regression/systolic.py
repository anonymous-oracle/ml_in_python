import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (20, 20)


df = pd.read_excel("../data_systolic.xls")

# # we will solve for x2 against x1 once and x3 against x1 separately, and both x2 and x3 against x1
# x = df.values
# plt.scatter(x[:,1], x[:,0])
# plt.show()
#
# plt.scatter(x[:,2], x[:,0])
# plt.show()

df['ones'] = 1
y = df['X1']
x = df[['X2', 'X3', 'ones']]
x2 = df[['X2', 'ones']]
x3 = df[['X3', 'ones']]

def get_r2(x, y):
    w = np.reshape(np.linalg.solve(x.T.dot(x), x.T.dot(y)), np.shape(x)[1])
    yhat = np.squeeze(x.dot(w))
    # model performance using r squared
    d1 = y - yhat
    d2 = y - y.mean()
    squared_sum_ratio = d1.dot(d1) / d2.dot(d2)

    return 1 - squared_sum_ratio

print("R^2 score with x2 only: {}".format(get_r2(x2, y)))
print("R^2 score with x3 only: {}".format(get_r2(x3, y)))
print("R^2 score with both x2 and x3: {}".format(get_r2(x, y)))
