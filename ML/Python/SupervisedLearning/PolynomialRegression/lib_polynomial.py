import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Data
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

# Polynomial Regression (degree 3)
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

# Predict a value
speed = mymodel(17)
print(f"Predicted value at x=17: {speed:.2f}")

# Generate line of best fit
x_line = numpy.linspace(min(x), max(x), 100)
y_line = mymodel(x_line)

# Plotting
plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x_line, y_line, color='red', label='Polynomial Fit (deg=3)')
# plt.axvline(x=17, linestyle='--', color='gray', label='x=17')
# plt.axhline(y=speed, linestyle='--', color='green', label=f'Predicted y={speed:.2f}')
plt.title('Polynomial Regression Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# R² Score
r2 = r2_score(y, mymodel(x))
print(f"R² Score: {r2:.4f}")
