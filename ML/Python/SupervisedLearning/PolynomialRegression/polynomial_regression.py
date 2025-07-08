import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

show = True

class PolynomialRegression():
    def __init__(self, Xtrain, Xtest, Ytrain, Ytest, alpha, degree):
        self.degree = degree
        self.alpha = alpha


        self.scaler = StandardScaler()
        self.Xtrain_raw = self.scaler.fit_transform(np.array(Xtrain).reshape(-1, 1))
        self.Xtest_raw = self.scaler.transform(np.array(Xtest).reshape(-1, 1))

        self.Ytrain = np.array(Ytrain).reshape(-1, 1)
        self.Ytest = np.array(Ytest).reshape(-1, 1)

        self.Xtrain = self.expand(self.Xtrain_raw)
        self.Xtest = self.expand(self.Xtest_raw)

        self.train_size = len(Xtrain)
        self.test_size = len(Xtest)
        self.theta = np.zeros((self.degree + 1, 1)) # Intercept, Linear Coefficent, Square term Coefficent

    def expand(self, x):
        return np.hstack([x**i for i in range(self.degree + 1)])
    
    def GradientDescent(self):
        pred = self.Xtrain.dot(self.theta)
        error = (pred - self.Ytrain)

        grad = (1 / self.train_size) * self.Xtrain.T.dot(error)
        self.theta -= self.alpha * grad

    def polynomial_regression(self):
        epoch = 1000

        for _ in range(epoch):
            self.GradientDescent()

    # R² = 1 - SS.tot / SS.res
    def R2Score(self, y_true, y_pred):
        ss_tot = np.sum((y_true - np.mean(y_pred))**2)
        ss_res = np.sum((y_true - y_pred)**2)

        r2 = 1 - (ss_res / ss_tot)

        print(f'Traning R² Score : {r2:.4f} ({r2 * 100:.2f}%)')

    def train(self):
        self.polynomial_regression()

        self.y_ = self.Xtrain.dot(self.theta)

        self.R2Score(self.Ytrain, self.y_)

        if show:
            plt.scatter(self.Xtrain_raw, self.Ytrain, color="blue", label="Train Data")
            x_line = np.linspace(self.Xtrain_raw.min(), self.Xtrain_raw.max(), 300).reshape(-1, 1)
            x_poly = self.expand(x_line)
            y_line = x_poly.dot(self.theta)
            plt.plot(x_line, y_line, color="red", label="Polynomial Fit")
            plt.legend()
            plt.title("Train Set Fit")
            plt.show()

    def test(self):
        self.y_ = self.Xtest.dot(self.theta)

        self.R2Score(self.Ytest, self.y_)

        if show:
            plt.scatter(self.Xtest_raw, self.Ytest, color="green", label="Test Data")
            x_line = np.linspace(self.Xtest_raw.min(), self.Xtest_raw.max(), 300).reshape(-1, 1)
            x_poly = self.expand(x_line)
            y_line = x_poly.dot(self.theta)
            plt.plot(x_line, y_line, color="red", label="Polynomial Fit")
            plt.legend()
            plt.title("Test Set Fit")
            plt.show()

def main():
    # x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
    # y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

    x = np.array([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22])
    y = np.array([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100])

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    pr = PolynomialRegression(Xtrain, Xtest, Ytrain, Ytest, alpha = 0.01, degree = 5)

    pr.train()
    pr.test()

if __name__ == "__main__":
    main()