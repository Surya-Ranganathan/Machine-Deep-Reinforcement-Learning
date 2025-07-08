import matplotlib.pyplot as plt
import numpy as np
import math

class KNearestNeighbor():
    def __init__(self, x, y, classes):
        self.x = x
        self.y = y
        self.classes = classes

    def Manhattan(self, ix, iy):
        distance = []
        for x_, y_, c_ in zip(self.x, self.y, self.classes):
            manhattan = abs(ix - x_) + abs(iy - y_)
            distance.append((manhattan, c_))
        
        distance.sort()
        print(distance)

    def Euclidean(self, ix, iy):
        distance = []
        for x_, y_, c_ in zip(self.x, self.y, self.classes):
            euclidean = math.sqrt((ix - x_)**2 + (iy - y_)**2)
            distance.append((euclidean, c_))
        
        distance.sort()
        print(distance)

    def Minkowski(self, ix, iy, p=3):
        distance = []
        for x_, y_, c_ in zip(self.x, self.y, self.classes):
            minkowski = (abs(ix - x_)**p + abs(iy - y_)**p)**(1/p)
            distance.append((minkowski, c_))
        
        distance.sort()
        print(f"\nMinkowski Distance (p={p}):")
        print(distance)

    def Cosine(self, ix, iy):
        distance = []
        for x_, y_, c_ in zip(self.x, self.y, self.classes):
            dot = ix * x_ + iy * y_
            norm1 = math.sqrt(ix**2 + iy**2)
            norm2 = math.sqrt(x_**2 + y_**2)
            cosine = 1 - (dot / (norm1 * norm2)) if norm1 and norm2 else 1
            distance.append((cosine, c_))

        distance.sort()
        print("\nCosine Distance:")
        print(distance)


def main():
    x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

    new_x = 8
    new_y = 21

    knn = KNearestNeighbor(x, y, classes)

    knn.Euclidean(new_x, new_y)
    knn.Manhattan(new_x, new_y)
    knn.Minkowski(new_x, new_y)
    knn.Cosine(new_x, new_y)


    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='blue' if classes[i]==0 else 'red')
        # plt.text(x[i]+0.3, y[i], f"C{classes[i]}")

    plt.scatter(new_x, new_y, c='green', marker='x', s=100, label='New Point')
    plt.legend()
    plt.title("KNN Classification (k=3)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()