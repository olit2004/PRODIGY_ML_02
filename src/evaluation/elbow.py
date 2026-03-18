from algorithms.kmeans_scratch import KMeansScratch
from evaluation.metrics import silhouette_score_scratch


def compute_elbow(X, max_k=10):

    inertias = []
    silhouettes = []

    for k in range(1, max_k + 1):

        model = KMeansScratch(n_clusters=k)
        model.fit(X)

        inertias.append(model.inertia_)

        if k > 1:
            score = silhouette_score_scratch(X, model.labels_, k)
        else:
            score = 0

        silhouettes.append(score)

    return inertias, silhouettes