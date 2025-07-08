import numpy as np
import matplotlib.pyplot as plt

show = True

class Gradient():
    def __init__(self, guess, alpha, iter):
        self.current = guess
        self. alpha = alpha
        self.iter = iter

    def function(self):
        return self.current * self.current

    def derivative(self):
        return 2 * self.current

    def gradient(self):
        
        x = []
        fx = []
        
        for i in range(0, self.iter):
            fx.append(self.function())
            self.current = self.current - self.alpha * self.derivative()
            x.append(self.current)

        return fx, x

def main():
    intial_guess = 5.0
    alpha = 0.1
    iter = 100

    gd = Gradient(intial_guess, alpha, iter)
    fx, x = gd.gradient()

    x_values = np.linspace(-5, 5, 100)
    y_values = x_values**2
    if (show):
        plt.plot(x_values, y_values, color = "blue", label = "f(x) = x^2")
        plt.plot(x[1:-1], fx[1:-1], label = "gradient descent point", color = "red", marker='o')

        plt.scatter(x[0], fx[0], color="green", label="Start (Initial Theta)")
        plt.scatter(x[-1], fx[-1], color="purple", label="End (Final Theta)")


        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Gradient Descent on the Parabola f(x) = x^2")
        plt.legend()

        plt.show()

if __name__ == "__main__":
    main()
