import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv("rgb_data.csv")

r = data["R"]
g = data["G"]
b = data["B"]

x = data[["R", "G", "B"]]

k = 27

km = KMeans(n_clusters = k, random_state = 0, n_init = "auto")
data["Cluster"] = km.fit_predict(x)

centroids = km.cluster_centers_

print("Cluster Centers:")

for i, centroid in enumerate(centroids): 
    
    print(f"Cluster {(i + 1)}: R = {centroid[0]:.2f}, G = {centroid[1]:.2f}, B = {centroid[2]:.2f}")

colors = data[["R", "G", "B"]] / 255.0

figure = plt.figure()
ax = figure.add_subplot(111, projection = "3d")

ax.scatter(r, g, b, c = colors, s = 40, alpha = 0.6)

ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c = "black", s = 150, marker = "x", label = "Centroids")

ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")

ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_zlim(0, 255)

ax.set_title("RGB Color Space With K-Means Clustering, K = 27")
ax.legend()

plt.show()