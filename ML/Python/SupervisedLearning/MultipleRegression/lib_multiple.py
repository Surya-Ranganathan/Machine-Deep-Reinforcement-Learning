import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data = {
    'Car': ['Toyota', 'Mitsubishi', 'Skoda', 'Fiat', 'Mini', 'VW', 'Skoda', 'Mercedes', 'Ford', 'Audi', 'Hyundai', 'Suzuki', 'Ford', 'Honda', 'Hundai', 'Opel', 'BMW', 'Mazda', 'Skoda', 'Ford', 'Ford', 'Opel', 'Mercedes', 'Skoda', 'Volvo', 'Mercedes', 'Audi', 'Audi', 'Volvo', 'BMW', 'Mercedes', 'Volvo', 'Ford', 'BMW', 'Opel', 'Mercedes'],
    'Model': ['Aygo', 'Space Star', 'Citigo', '500', 'Cooper', 'Up!', 'Fabia', 'A-Class', 'Fiesta', 'A1', 'I20', 'Swift', 'Fiesta', 'Civic', 'I30', 'Astra', '1', '3', 'Rapid', 'Focus', 'Mondeo', 'Insignia', 'C-Class', 'Octavia', 'S60', 'CLA', 'A4', 'A6', 'V70', '5', 'E-Class', 'XC70', 'B-Max', '2', 'Zafira', 'SLK'],
    'Volume': [1000, 1200, 1000, 900, 1500, 1000, 1400, 1500, 1500, 1600, 1100, 1300, 1000, 1600, 1600, 1600, 1600, 2200, 1600, 2000, 1600, 2000, 2100, 1600, 2000, 1500, 2000, 2000, 1600, 2000, 2100, 2000, 1600, 1600, 1600, 2500],
    'Weight': [790, 1160, 929, 865, 1140, 929, 1109, 1365, 1112, 1150, 980, 990, 1112, 1252, 1326, 1330, 1365, 1280, 1119, 1328, 1584, 1428, 1365, 1415, 1415, 1465, 1490, 1725, 1523, 1705, 1605, 1746, 1235, 1390, 1405, 1395],
    'CO2': [99, 95, 95, 90, 105, 105, 90, 92, 98, 99, 99, 101, 99, 94, 97, 97, 99, 104, 104, 105, 94, 99, 99, 99, 99, 102, 104, 114, 109, 114, 115, 117, 104, 108, 109, 120]
}

df = pd.DataFrame(data)

x = df[["Volume", "Weight"]].values
y = df["CO2"].values

X = df[['Weight', 'Volume']]
y = df['CO2']

# Train model
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Predict CO2 for new data
predictedCO2 = regr.predict([[2300, 1300]])
print(f"Predicted CO2 for weight=2300kg and volume=1300cm³: {predictedCO2[0]:.2f} g/km")

# R² Score
y_pred = regr.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² score: {r2:.4f} ({r2 * 100:.2f}%)")

# Plot actual vs predicted
plt.scatter(y, y_pred, color='blue', label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual CO₂ Emissions")
plt.ylabel("Predicted CO₂ Emissions")
plt.title("Actual vs Predicted CO₂ Emissions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()