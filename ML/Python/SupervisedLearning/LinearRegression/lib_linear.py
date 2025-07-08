import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

# Data
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Linear regression
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(f'final outcomes :\nslope : {slope}\nintercept : {intercept}')


# Prediction function
def myfunc(x):
    return slope * x + intercept

# Predicted values
mymodel = list(map(myfunc, x))

# Accuracy (R²)
r2 = r2_score(y, mymodel) * 100
print(f"R² Score (Accuracy): {r2:.4f}")

# Plot
plt.scatter(x, y, label='Data Points')
plt.plot(x, mymodel, color='red', label='Regression Line')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()
