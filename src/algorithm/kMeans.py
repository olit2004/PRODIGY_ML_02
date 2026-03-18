import numpy as np



class KMeansScratch:
    """
    K-Means clustering implementation with K-Means++ initialization.
    """

    def __init__(self, n_clusters=5, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters  # the number of clusters we want 

        self.max_iters = max_iters   
        self.tol = tol # our toleration level
        self.random_state = random_state  # ensures reproducibility 

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0



    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            distances = np.array(
                [min(np.linalg.norm(x - c) for c in centroids) for x in X]
            )

            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)

            r = np.random.rand()
            for idx, p in enumerate(cumulative_probs):
                if r < p:
                    centroids.append(X[idx])
                    break
        return np.array(centroids)


    
    def _assign_clusters(self, X):
        labels = np.zeros(len(X), dtype=int)
        distances = np.zeros(len(X))

        for i in range(len(X)):
            dists = [np.linalg.norm(X[i] - c) for c in self.centroids]
            labels[i] = np.argmin(dists)
            distances[i] = dists[labels[i]]

        return labels, distances

    
    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                new_centroids[k] = X[np.random.randint(X.shape[0])]

        return new_centroids
    

    
    def _has_converged(self, old, new):
        shift = np.max([np.linalg.norm(old[i] - new[i]) for i in range(self.n_clusters)])
        return shift < self.tol

    
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iters):

            labels, _ = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)

            if self._has_converged(self.centroids, new_centroids):
                self.centroids = new_centroids
                self.n_iter_ = i + 1
                break

            self.centroids = new_centroids
            self.n_iter_ = i + 1

        self.labels_, distances = self._assign_clusters(X)
        self.inertia_ = np.sum(distances ** 2)

        return self

    
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction.")

        labels, _ = self._assign_clusters(X)
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_