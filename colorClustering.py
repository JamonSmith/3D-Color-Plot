import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

RED = "\033[31m"
RESET = "\033[0m"

def on_key(event):
    
    if event.key == '0':
        
        plt.close()
        

def getBin(val):
    
    if val > 171:
    
        return "H"
        
    elif val > 86:
    
        return "M"
        
    else:
    
        return "L"
        
def nameColor(r, g, b):

    label = getBin(r) + getBin(g) + getBin(b)
    
    colorNames = {
        
        "LLL": "Black/Dark",
        "LLM": "Dark Blue",
        "LLH": "Blue",
        
        "LML": "Dark Green",
        "LMM": "Dark Cyan",
        "LMH": "Light Blue",
        
        "LHL": "Green",
        "LHM": "Mint Green",
        "LHH": "Cyan",
        
        "MLL": "Maroon",
        "MLM": "Dark Purple",
        "MLH": "Purple",
        
        "MML": "Olive",
        "MMM": "Grey",
        "MMH": "Periwinkle",
        
        "MHL": "Volt Yellow",
        "MHM": "Light Green",
        "MHH": "Frosty Blue",
        
        "HLL": "Red", 
        "HLM": "Rosy Pink",
        "HLH": "Magenta",
        
        "HML": "Orange",
        "HMM": "Light Red",
        "HMH": "Pink/Purple",
        
        "HHL": "Yellow",
        "HHM": "Light Yellow",
        "HHH": "White/Light"
    }

    return colorNames[label]
    
print(RED + "Press 0 to close the visual" + RESET)
print()  

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
    
    colorName = nameColor(centroid[0], centroid[1], centroid[2])
    
    print(
        f"Cluster {i + 1}: \t{colorName} "
        f"(R = {centroid[0]:.2f}, G = {centroid[1]:.2f}, B = {centroid[2]:.2f})"
    )

colors = data[["R", "G", "B"]] / 255.0

figure = plt.figure()
ax = figure.add_subplot(111, projection = "3d")

ax.scatter(r, g, b, c = colors, s = 10, alpha = 0.4)

ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c = "black", s = 150, marker = "x", label = "Centroids")

for i, centroid in enumerate(centroids): 
    
    colorName = nameColor(centroid[0], centroid[1], centroid[2])
    
    ax.text(centroid[0] + 5, centroid[1] + 5, centroid[2] + 5, colorName, fontsize = 8)

ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")

ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_zlim(0, 255)

ax.set_title("RGB Color Space With K-Means Clustering, K = 27")
ax.legend()

figure.canvas.mpl_connect('key_press_event', on_key)        

plt.show()