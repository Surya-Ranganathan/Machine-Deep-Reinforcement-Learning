import numpy as np
import matplotlib.pyplot as plt

m = -0.3654
c = 95.3556

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)
plt.plot(x, (m*x)+c)

plt.xlabel("X data")
plt.ylabel("y data")

plt.title("Teat Data")

plt.show()