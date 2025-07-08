import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("advertising.csv")

plt.subplot(2,2,1)
plt.plot(np.array(df.TV))
plt.title("TV")

plt.subplot(2,2,2)
plt.plot(np.array(df.Radio))
plt.title("Radio")

plt.subplot(2,2,3)
plt.plot(np.array(df.Newspaper))
plt.title("Newspaper")

plt.subplot(2,2,4)
plt.plot(np.array(df.Sales))
plt.title("Sales")

plt.suptitle("Advertising")

plt.show()
