import numpy as np


def silhouette_score_scratch(X, labels, n_clusters):

    n_samples = X.shape[0]
    scores = []

    for i in range(n_samples):

        current_cluster = labels[i]

        same_cluster = X[labels == current_cluster]

        if len(same_cluster) > 1:
            a = np.mean(
                [
                    np.linalg.norm(X[i] - point)
                    for point in same_cluster
                    if not np.array_equal(point, X[i])
                ]
            )
        else:
            a = 0

        b = np.inf

        for k in range(n_clusters):
            if k == current_cluster:
                continue

            other_cluster = X[labels == k]

            if len(other_cluster) > 0:
                dist = np.mean([np.linalg.norm(X[i] - p) for p in other_cluster])
                b = min(b, dist)

        score = (b - a) / max(a, b) if max(a, b) > 0 else 0
        scores.append(score)

    return np.mean(scores)