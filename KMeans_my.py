import numpy as np

class KMeans_my:
    def __init__(self, k=5, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        # Randomly initialize centroids
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iter):
            # Calculate Euclidian distance
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

            # Assign labels based on the nearest centroid
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            # Convergence - when there is no change in new centroids and previous values of centroids.
            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids
        
        self.labels_ = labels
        self.cluster_centers_ = centroids

        return self.labels_, self.cluster_centers_