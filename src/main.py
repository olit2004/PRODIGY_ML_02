from utils.data_loader import load_customer_data
from algorithms.kmeans_scratch import KMeansScratch
from evaluation.elbow import compute_elbow
from visualization.plots import plot_elbow, plot_clusters


def run_pipeline():

    df, X = load_customer_data()

    inertias, silhouettes = compute_elbow(X)

    plot_elbow(inertias, silhouettes)

    kmeans = KMeansScratch(n_clusters=5)

    labels = kmeans.fit_predict(X)

    plot_clusters(X, labels, kmeans.centroids)


if __name__ == "__main__":
      run_pipeline()