import numpy as np
from datetime import datetime

a = np.random.randn(100)
b = np.random.randn(100)
T = 10000
def slow_dot_product(a, b):
    result = 0
    for e, f in zip(a, b):
        result += e*f
    return result

runs = []
for i in range(20):
    t0 = datetime.now()
    for t in range(T):
        slow_dot_product(a, b)
    dt1 = datetime.now() - t0

    t0 = datetime.now()
    for t in range(T):
        a.dot(b)
    dt2 = datetime.now() - t0

    runs.append(dt1.total_seconds() / dt2.total_seconds())

print("average dt1 / dt2: ", np.average(runs))