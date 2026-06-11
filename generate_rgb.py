import pandas as pd
import numpy as np

numPts = 1000

rgbVals = np.random.randint(0, 256, size = (numPts, 3))

df = pd.DataFrame(rgbVals, columns = ["R", "G", "B"])

df.to_csv("rgb_data.csv", index = False)