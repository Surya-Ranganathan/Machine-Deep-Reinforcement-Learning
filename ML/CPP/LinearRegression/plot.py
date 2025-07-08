import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m = -1.60916
c = 101.727

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y, color = 'blue')
plt.plot(x, (m * x) + c, color = 'red')
plt.title("Random Data")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()

plt.show()