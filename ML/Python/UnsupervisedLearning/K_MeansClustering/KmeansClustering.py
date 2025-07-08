import matplotlib.pyplot as plt
import numpy as np
import math
import copy as cp

show = True

class KMeansClustering():
    def __init__(self, x, y, nCluster):
        self.x = x
        self.y = y
        self.nCluster = nCluster
        self.size = len(x)
        self.euclidean = []

    def UpdateCenteroids(self):
        for i in range(len(self.centroids)):
            mean_x = np.mean([x for x, y, e in zip(self.x, self.y, self.euclidean) if e['cluster'] == f'c{i}'])
            mean_y = np.mean([y for x, y, e in zip(self.x, self.y, self.euclidean) if e['cluster'] == f'c{i}'])

            self.centroids[i][0] = mean_x
            self.centroids[i][1] = mean_y


    def EuclideanDistance(self):
        self.euclidean = []

        for ix, iy in zip(self.x, self.y):
            min = float('inf')
            clust = ''

            for size, (cx, cy) in enumerate(self.centroids):
                dist = math.sqrt((cx - ix) ** 2 + (cy - iy) ** 2)
                
                if dist < min:
                    min = dist
                    clust = f'c{size}'

            self.euclidean.append({'dist': min, 'cluster': clust})

        for i in self.euclidean:
            print(i)

    def K_means_clustering(self, epoch):
        rand_index = np.random.choice(self.size, self.nCluster, replace = "False")
        print(f'randon number of {self.nCluster} : {rand_index}')

        self.centroids = self.oldCenteroids = [[self.x[i], self.y[i]] for i in rand_index]

        print(f'randon number of {rand_index} : {self.centroids}')

        for ep in range(epoch):
            print(f'Epoch {ep + 1} Centroids: {self.centroids}')
            
            self.EuclideanDistance()
            oldCenteroids = cp.deepcopy(self.centroids)
            self.UpdateCenteroids()

            # if show:
            #     plt.figure()
            #     for i in range(len(self.centroids)):
            #         dataX = [x for x, y, e in zip(self.x, self.y, self.euclidean) if e['cluster'] == f'c{i}']
            #         dataY = [y for x, y, e in zip(self.x, self.y, self.euclidean) if e['cluster'] == f'c{i}']
            #         plt.scatter(dataX, dataY, label=f'c{i}')
                
            #     # Plot centroids
            #     centroidX = [c[0] for c in self.centroids]
            #     centroidY = [c[1] for c in self.centroids]
            #     plt.scatter(centroidX, centroidY, c='black', marker='x', s=100, label='Centroids')

            #     plt.title(f'Iteration {ep + 1}')
            #     plt.legend()
            #     plt.grid(True)
            #     plt.show()
            
            if oldCenteroids == self.centroids:
                break

            if show:
                plt.figure()
                for i in range(len(self.centroids)):
                    dataX = [x for x, y, e in zip(self.x, self.y, self.euclidean) if e['cluster'] == f'c{i}']
                    dataY = [y for x, y, e in zip(self.x, self.y, self.euclidean) if e['cluster'] == f'c{i}']
                    plt.scatter(dataX, dataY, label=f'c{i}')
                
                # Plot centroids
                centroidX = [c[0] for c in self.centroids]
                centroidY = [c[1] for c in self.centroids]
                plt.scatter(centroidX, centroidY, c='black', marker='x', s=100, label='Centroids')

                plt.title(f'Iteration {ep + 1}')
                plt.legend()
                plt.grid(True)
                plt.show()

def main():
    x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

    kmc = KMeansClustering(x, y, nCluster = 4)

    kmc.K_means_clustering(epoch = 100)

if __name__ == "__main__":
    main()