import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ANSI ESCAPE CODES
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
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

def loadData(fileName):
    
    data = pd.read_csv(fileName) 
    x = data[["R", "G", "B"]]
    
    return data, x

def kMeans(data, x, k):

    km = KMeans(n_clusters = k, random_state = 0, n_init = "auto")
    data["Cluster"] = km.fit_predict(x)
    
    centroids = km.cluster_centers_
    
    return km, centroids

def printCentroids(centroids):

    print("Cluster Centers:")
    print()

    for i, centroid in enumerate(centroids): 
        
        colorName = nameColor(centroid[0], centroid[1], centroid[2])
        
        print(
            f"\tCluster {i + 1}: \t{colorName} "
            f"(R = {centroid[0]:.2f}, "
            f"G = {centroid[1]:.2f}, "
            f"B = {centroid[2]:.2f})"
        )

def plotData(data, centroids, k):

    print("Press 0 to close the visual")
    print()
    
    r = data["R"]
    g = data["G"]
    b = data["B"]
    
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
    
    ax.set_title(f"RGB Color Space With K-Means Clustering, K = {k}")
    ax.legend()
    
    figure.canvas.mpl_connect('key_press_event', on_key)        
    
    plt.show()
    
def classifyColor(km, centroids):

    print()
    print("Enter RGB values:")
    
    rComponent = int(input("Enter Red Component (0-255): "))
    gComponent = int(input("Enter Green Component (0-255): "))
    bComponent = int(input("Enter Blue Component (0-255): "))
    
    userColor = [[rComponent, gComponent, bComponent]]
    
    index = km.predict(userColor)[0]
    centroid = centroids[index]
    
    color = nameColor(centroid[0], centroid[1], centroid[2])
    print()
    
    print(f"Your color belongs to Cluster {index + 1}: {color}")
    print()
    print(
        f"Nearest centroid: "
        f"R = {centroid[0]:.2f}, "
        f"G = {centroid[1]:.2f}, "
        f"B = {centroid[2]:.2f})"
    )

def main():
    
    print() 
    print(GREEN + "Jamon Smith" + RESET)   
    print()  
    
    file = "rgb_data.csv"
    k = 27
    
    data, x = loadData(file)
    
    km, centroids = kMeans(data, x, k)
    
    while True:
    
        print()
        print("========== RGB Color Clustering ==========")
        print()
        print("1.) View Cluster Centroids")
        print("2.) View 3D Visualization")
        print("3.) Classify a Color")
        print("0.) Exit Program")
        print()
        
        userInput = int(input("Choose a command; 1, 2, 3, or 0: "))
        print()
        
        if userInput == 1:
    
            printCentroids(centroids)
    
        elif userInput == 2:
        
            plotData(data, centroids, k)
    
        elif userInput == 3:
            
            classifyColor(km, centroids)
    
        elif userInput == 0:
        
            print("Thank you, goodbye!")
            break
            
        else:
            
            print("Invalid command, choose either 1, 2, 3, or 0")
            
if __name__ == "__main__":
    main()