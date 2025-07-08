import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data points
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(data)

print(kmeans.cluster_centers_)

# Plotting
plt.scatter(x, y, c=kmeans.labels_, cmap='viridis')
plt.scatter(*zip(*kmeans.cluster_centers_), c='red', marker='x', s=100, label='Centroids')
plt.title("K-Means Clustering (k=2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
