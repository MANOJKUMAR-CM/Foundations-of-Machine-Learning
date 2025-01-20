import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("cm_dataset_2.csv", names=['X','Y'])
print(data.head())

Data = data.values

class Kmeans:
    def __init__(self, k=2):
        self.k = k
        self.centroids = None
        self.label = None
        
    def fit(self, X):
        index = np.random.choice(X.shape[0], self.k) # selecting k random index out of 1000 to be cluster centroids
        self.centroids = X[index] # Points @ the index obtained in prev step
        
        Error = []
        for i in range(100):
             # Assigning points to cluster based on euclidean distance
            d = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2) # computing the distance of each point with each of the cluster centroid
            self.label = np.argmin(d, axis=1) # assigning point to the cluster whose distance with respect to mean is minimum
            
            self.centroids = np.array([X[self.label == i].mean(axis=0) for i in range(self.k)]) # Updating the mean of the centroids
            error = np.sum((X - self.centroids[self.label])**2)
            
            Error.append(error)
            if(len(Error)>1 and (Error[-2] == Error[-1])):
                break
        return  self.centroids,Error, self.label
    
    def predict(self, X):
        if(self.centroids is None):
            raise ValueError("Model has not fitted yet")
        else:
            d = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            label = np.argmin(d,axis=1)
            return label



# 5 Initialization and their corresponding errors and clusters obtained
for i in range(5):
    centroids, error, label = Kmeans().fit(Data)
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].scatter(Data[:, 0], Data[:, 1], c=label, cmap='plasma', marker='o', alpha=0.5, )
    ax[0].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label="Cluster Means")
    ax[0].set_title(f"Clusters and Centroids (Iteration {i+1})")
    ax[0].grid(True)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend()

    ax[1].plot(error, marker='o')
    ax[1].grid(True)
    ax[1].set_title(f"Error Curve Iteration{i+1}")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Error")
    
    plt.tight_layout()
    plt.show()
    
    
def VoronoiRegion(model, Data):
    x_min, x_max = Data[:, 0].min() - 1, Data[:, 0].max() + 1
    y_min, y_max = Data[:, 1].min() - 1, Data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    label_pred = model.predict(grid_points).reshape(xx.shape)
    
    # Plot the Voronoi regions
    plt.contourf(xx, yy, label_pred, cmap="viridis", alpha=0.5)
    plt.contour(xx, yy, label_pred, colors='k', linewidths=0.2)
    
    # Plot the data points
    plt.scatter(Data[:, 0], Data[:, 1], c=model.label, cmap="plasma", edgecolor='k', s=30, alpha=0.5)
    
    # Plot the centroids
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', s=100, marker='X', edgecolor='k', label="Cluster Mean")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f'Voronoi Regions for K={model.k}')
    plt.tight_layout()
    plt.show()

for k in [2,3,4,5]:
    K = Kmeans(k)
    centroids, error, label = K.fit(Data)
    print(f"For K = {k} the cluster mean is:")
    print(centroids)
    print()
    VoronoiRegion(K, Data)
    

