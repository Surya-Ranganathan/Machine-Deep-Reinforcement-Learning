import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('output.csv')

theta_values = data['theta'].values
function_values = data['f(theta)'].values

x_values = np.linspace(-5, 5, 100)
y_values = x_values**2

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="f(x) = x^2", color="blue")

plt.plot(theta_values, function_values, label="Gradient Descent Path", color="red", marker='o')

plt.scatter(theta_values[0], function_values[0], color="green", zorder=5, label="Start (Initial Theta)")
plt.scatter(theta_values[-1], function_values[-1], color="purple", zorder=5, label="End (Final Theta)")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent on the Parabola f(x) = x^2")
plt.legend()

plt.grid(True)
plt.show()
