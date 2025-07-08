import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

show = True

class LinearRegression():
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest, alpha):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest
        self.slope = 0.0
        self.intercept = 0.0
        self.train_size = len(Xtrain)
        self.test_size = len(Xtest)
        self.alpha = alpha
        self.mse = 0.0

    def GradientDescent(self):
        dm = 0.0
        dc = 0.0
        self.mse = 0.0
        ypred = []

        for i in range(self.train_size):
            pred = self.slope * self.Xtrain[i] + self.intercept 
            error = pred - self.Ytrain[i]

            # self.mse += error * error
            
            dm += error * self.Xtrain[i]
            dc += error

            ypred.append(pred)

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
    def R2Score(self, dataX, dataY):
        ss_tot = sum((yi - np.mean(dataY))**2 for yi in dataY)
        ss_res = sum((yi - yip)**2 for yi,yip in zip(dataY,self.y_))

        r2 = 1 - (ss_res / ss_tot)

        print(f'Traning R² Score : {r2:.4f} ({r2 * 100:.2f}%)')

    def train(self):
        self.linear_regression()
        print(f'final outcomes :\nslope : {self.slope}\nintercept : {self.intercept}\nmse : {self.mse}')

        self.y_ = [self.slope * xi + self.intercept for xi in self.Xtrain]

        self.R2Score(self.Xtrain, self.Ytrain)

        if (show):
            plt.scatter(self.Xtrain, self.Ytrain, color = "blue", label = "Train data")
            plt.plot(self.Xtrain, self.y_, color = "red", label = "Best fit" )
            plt.legend()
            plt.show()

    def test(self):
        self.y_ = [self.slope * xi + self.intercept for xi in self.Xtest]

        self.R2Score(self.Xtest, self.Ytest)

        plt.scatter(self.Xtest, self.Ytest, color = 'blue')
        plt.plot(self.Xtest, self.y_, color = 'red')

        plt.show()

def main():
    x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
    y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

    lr = LinearRegression(Xtrain, Ytrain, Xtest, Ytest, alpha = 0.01)

    lr.train()
    lr.test()

if __name__ == "__main__":
    main()