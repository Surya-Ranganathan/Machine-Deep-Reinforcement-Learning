import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

show = True

class MultipleRegression():
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest, alpha):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain.reshape(-1, 1)
        self.Xtest  = Xtest
        self.Ytest  = Ytest.reshape(-1, 1)
        self.alpha = alpha

        self.n_features = Xtrain.shape[1]  # FIXED
        self.slope = np.zeros((self.n_features, 1))
        self.intercept = 0.0

        self.train_size = len(Xtrain) 
        self.test_size = len(Xtest)

    def GradientDescent(self):
        pred = np.dot(self.Xtrain, self.slope) + self.intercept
        error = pred - self.Ytrain

        dm = np.dot(self.Xtrain.T, error)
        dc = np.sum(error)

        dm *= (2.0 / self.train_size)
        dc *= (2.0 / self.train_size)

        self.slope -= self.alpha * dm
        self.intercept -= self.alpha * dc

    def linear_regression(self):
        Epoch = 50000

        while Epoch:
            self.GradientDescent()
            Epoch -= 1
    
    # R² = 1 - SS.tot / SS.res
    def R2Score(self, y_true, y_pred):
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"R²: {r2:.4f} ({r2 * 100:.2f}%)")

    def train(self):
        self.linear_regression()
        print(f'final outcomes :\nslope : {self.slope}\nintercept : {self.intercept}')

        self.y_ = np.dot(self.Xtrain, self.slope) + self.intercept

        self.R2Score(self.Ytrain, self.y_)

        if show:
            plt.scatter(self.Ytrain, self.y_, color="blue", label="Train Data")
            plt.plot(self.Ytrain, self.Ytrain, color="red", label="Perfect Fit")
            plt.legend()
            plt.xlabel("Actual CO2")
            plt.ylabel("Predicted CO2")
            plt.title("Train Set Prediction vs Actual")
            plt.show()

    def test(self):
        self.y_ = np.dot(self.Xtest, self.slope) + self.intercept

        self.R2Score(self.Ytest, self.y_)

        if show:
            plt.scatter(self.Ytest, self.y_, color='green', label="Test Data")
            plt.plot(self.Ytest, self.Ytest, color='red', label="Perfect Fit")
            plt.xlabel("Actual CO2")
            plt.ylabel("Predicted CO2")
            plt.title("Test Set Prediction vs Actual")
            plt.legend()
            plt.show()

def main():
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

    lr = MultipleRegression(Xtrain, Ytrain, Xtest, Ytest, alpha = 0.01)

    lr.train()
    lr.test()

if __name__ == "__main__":
    main()