from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Centralized output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_elbow(inertias, silhouettes, save=True, show=False):
    """
    Plot Elbow (Inertia) and Silhouette scores.

    Parameters:
    -----------
    inertias : list
    silhouettes : list
    save : bool → save image to disk
    show : bool → display plot
    """

    k_range = range(1, len(inertias) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow plot
    axes[0].plot(k_range, inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")

    # Silhouette plot
    axes[1].plot(k_range, silhouettes, marker="s")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Score")

    plt.tight_layout()

    if save:
        file_path = OUTPUT_DIR / "elbow_silhouette.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_clusters(X, labels, centroids, save=True, show=False):
    """
    Plot clustered data with centroids.

    Parameters:
    -----------
    X : np.ndarray
    labels : np.ndarray
    centroids : np.ndarray
    save : bool
    show : bool
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        s=50,
        alpha=0.8
    )

    # Centroids
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=250,
        color="red",
        label="Centroids"
    )

    ax.set_title("Customer Segmentation (K-Means)")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1–100)")

    ax.legend()

    # Optional colorbar (helps interpretation)
    plt.colorbar(scatter, ax=ax, label="Cluster")

    plt.tight_layout()

    if save:
        file_path = OUTPUT_DIR / "clusters.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)