import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("rgb_data.csv")
 
print(data)

r = data["R"]
g = data["G"]
b = data["B"]

colors = data[["R", "G", "B"]] / 255.0

figure = plt.figure()
ax = figure.add_subplot(111, projection = "3d")

ax.scatter(r, g, b, c = colors, s = 40)

ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")

ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_zlim(0, 255)

ax.set_title("RGB Color Space")

plt.show()