import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

show = True

class LogisticRegression():
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest, alpha):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.alpha = alpha
        self.m = self.Xtrain.shape[0]
        self.n = self.Xtrain.shape[1]
        self.weight = np.zeros((self.n, 1))
        self.bias = 0.0

    def Sigmoid(self, x):
        z = np.dot(x, self.weight) + self.bias

        return 1 / (1 + np.exp(-z))

    def GradientDescent(self):
        sigmoid = self.Sigmoid(self.Xtrain)

        cost = -np.mean(self.Ytrain * np.log(sigmoid + 1e-9) +
                        (1 - self.Ytrain) * np.log(1 - sigmoid + 1e-9))

        dw = (1 / self.m) * np.dot(self.Xtrain.T, (sigmoid - self.Ytrain))
        db = (1 / self.m) * np.sum(sigmoid - self.Ytrain)

        self.weight -= self.alpha * dw
        self.bias -= self.alpha * db

    def logistic_regression(self):
        epochs=1000

        for _ in range(epochs):
            self.GradientDescent()

    def predict(self, X):
        return (self.Sigmoid(X) >= 0.5).astype(int)

    def accuracy(self, y_true, y_pred):
        acc = np.mean(y_true == y_pred) * 100

        print(f"Accuracy: {acc:.2f}%")

    def plot_decision_boundary(self, X, y, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = self.predict(grid).reshape(xx.shape)

        plt.contourf(xx, yy, probs, alpha=0.3, cmap='bwr')
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()

    def train(self):
        self.logistic_regression()
        y_pred = self.predict(self.Xtrain)
        self.accuracy(self.Ytrain, y_pred)

        if show:
            self.plot_decision_boundary(self.Xtrain, self.Ytrain, title="Train Set Decision Boundary")

    def test(self):
        y_pred = self.predict(self.Xtest)
        self.accuracy(self.Ytest, y_pred)

        if show:
            self.plot_decision_boundary(self.Xtest, self.Ytest, title="Test Set Decision Boundary")

def main():
    x = np.array([
        [34, 78], [45, 85], [50, 43], [61, 70], [71, 80],
        [60, 52], [75, 89], [55, 42], [80, 90], [52, 65],
        [47, 50], [33, 38], [87, 96], [65, 70], [50, 35]
    ])

    y = np.array([
        0, 0, 0, 1, 1,
        1, 1, 0, 1, 1,
        0, 0, 1, 1, 0
    ]).reshape(-1, 1)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    lr = LogisticRegression(Xtrain, Ytrain, Xtest, Ytest, alpha=0.1)

    lr.train()
    lr.test()

if __name__ == "__main__":
    main()
