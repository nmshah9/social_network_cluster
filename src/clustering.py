"""
clustering.py
-------------
Implements and compares three clustering algorithms -- K-Means,
Agglomerative (Hierarchical), and DBSCAN -- on the scaled student feature
matrix, plus helper functions for choosing k, plotting a dendrogram, and
building demographic / trend profiles per cluster.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from src import config


# ---------------------------------------------------------------------------
# K-Means: choosing k
# ---------------------------------------------------------------------------
def kmeans_elbow_and_silhouette(X: np.ndarray, k_range=range(2, 11), sample_size: int = 3000):
    """
    Fits K-Means for each k in k_range and records inertia (for the elbow
    plot) and silhouette score. Returns a tidy DataFrame. Silhouette is
    computed on a random subsample for speed/memory on large datasets
    (silhouette is O(n^2)).
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sil = _silhouette_on_sample(X, labels, sample_size=sample_size)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def plot_elbow_and_silhouette(scores_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(scores_df["k"], scores_df["inertia"], marker="o", color="#4C72B0")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (within-cluster SSE)")
    axes[0].set_title("Elbow method")

    axes[1].plot(scores_df["k"], scores_df["silhouette"], marker="o", color="#55A868")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score vs k")
    fig.tight_layout()
    return fig


def fit_kmeans(X: np.ndarray, k: int):
    km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


# ---------------------------------------------------------------------------
# Hierarchical / Agglomerative
# ---------------------------------------------------------------------------
def plot_dendrogram(X: np.ndarray, sample_size: int = 500, method: str = "ward"):
    """
    Full-data dendrograms are unreadable (and slow) for thousands of rows,
    so we plot on a random subsample purely to visualise the linkage
    structure and pick a sensible number of clusters.
    """
    rng = np.random.RandomState(config.RANDOM_STATE)
    if X.shape[0] > sample_size:
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    Z = linkage(X_sample, method=method)
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_rotation=90)
    ax.set_title(f"Hierarchical clustering dendrogram (subsample of {len(X_sample)}, "
                 f"'{method}' linkage, truncated)")
    ax.set_xlabel("Cluster size (leaf index)")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig


def fit_hierarchical(X: np.ndarray, n_clusters: int, linkage_method: str = "ward"):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = agg.fit_predict(X)
    return agg, labels


def hierarchical_silhouette_by_k(X: np.ndarray, k_range=range(2, 11), linkage_method="ward",
                                  sample_size: int = 3000):
    rows = []
    for k in k_range:
        _, labels = fit_hierarchical(X, k, linkage_method)
        sil = _silhouette_on_sample(X, labels, sample_size=sample_size)
        rows.append({"k": k, "silhouette": sil})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
def plot_k_distance(X: np.ndarray, k: int = 5):
    """
    k-distance plot to help pick DBSCAN's `eps`: the "elbow" in the sorted
    k-th nearest-neighbour distance curve is a good starting eps.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_dist, color="#C44E52")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"{k}-th nearest neighbour distance")
    ax.set_title("K-distance plot (look for the 'elbow' -> good eps)")
    fig.tight_layout()
    return fig


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int = 5):
    # algorithm='ball_tree' avoids sklearn falling back to a dense O(n^2)
    # pairwise distance matrix in higher-dimensional feature spaces, which
    # is what was causing out-of-memory kills on the full 15k-row dataset.
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1)
    labels = db.fit_predict(X)
    return db, labels


def _silhouette_on_sample(X, labels, sample_size=3000, random_state=config.RANDOM_STATE):
    """Silhouette score is O(n^2) memory/time -- evaluate on a random
    subsample for large datasets instead of the full set."""
    mask = labels != -1
    X_valid, labels_valid = X[mask], labels[mask]
    if len(set(labels_valid)) < 2:
        return np.nan
    if len(X_valid) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_valid), sample_size, replace=False)
        X_valid, labels_valid = X_valid[idx], labels_valid[idx]
    return silhouette_score(X_valid, labels_valid)


def dbscan_grid_search(X: np.ndarray, eps_values, min_samples_values, sample_size=3000):
    """
    Try combinations of eps / min_samples. Only combinations that yield
    >= 2 real clusters (excluding noise label -1) get a silhouette score;
    noise points are excluded from the silhouette calculation since -1 isn't
    a real cluster. Silhouette is computed on a random subsample for speed
    on large datasets.
    """
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            _, labels = fit_dbscan(X, eps, min_samples)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = round(100 * np.mean(labels == -1), 2)
            sil = np.nan
            if n_clusters >= 2:
                sil = _silhouette_on_sample(X, labels, sample_size=sample_size)
            rows.append({"eps": eps, "min_samples": min_samples,
                         "n_clusters": n_clusters, "noise_pct": noise_pct,
                         "silhouette": sil})
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False)


# ---------------------------------------------------------------------------
# Comparison across the 3 algorithms
# ---------------------------------------------------------------------------
def compare_algorithms(X: np.ndarray, kmeans_labels, hier_labels, dbscan_labels,
                        sample_size: int = 3000):
    """
    Builds a comparison table of silhouette scores for the 3 fitted
    clusterings. For DBSCAN, noise points (-1) are excluded from the score
    since silhouette isn't defined for a "no cluster" label. Silhouette is
    computed on a shared random subsample so the comparison is both fair
    and fast on large datasets (silhouette is O(n^2)).
    """
    results = []

    results.append({"algorithm": "K-Means",
                     "n_clusters": len(set(kmeans_labels)),
                     "silhouette": _silhouette_on_sample(X, kmeans_labels, sample_size)})

    results.append({"algorithm": "Hierarchical (Agglomerative)",
                     "n_clusters": len(set(hier_labels)),
                     "silhouette": _silhouette_on_sample(X, hier_labels, sample_size)})

    n_db_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    db_sil = (_silhouette_on_sample(X, dbscan_labels, sample_size)
              if n_db_clusters >= 2 else np.nan)
    results.append({"algorithm": "DBSCAN",
                     "n_clusters": n_db_clusters,
                     "silhouette": db_sil})

    return pd.DataFrame(results).sort_values("silhouette", ascending=False)


def plot_algorithm_comparison(comparison_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_df = comparison_df.dropna(subset=["silhouette"])
    ax.bar(plot_df["algorithm"], plot_df["silhouette"],
           color=["#4C72B0", "#55A868", "#C44E52"][:len(plot_df)])
    ax.set_ylabel("Silhouette score")
    ax.set_title("Clustering algorithm comparison")
    for i, v in enumerate(plot_df["silhouette"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2-D visualisation via PCA
# ---------------------------------------------------------------------------
def pca_2d(X: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(X)
    return coords, pca


def plot_clusters_2d(coords: np.ndarray, labels, title: str = "Clusters (PCA projection)"):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                          s=12, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster",
                         bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.add_artist(legend1)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demographic profiling / trend analysis per cluster
# ---------------------------------------------------------------------------
def build_cluster_profile(df_with_clusters: pd.DataFrame, cluster_col: str = "cluster"):
    """
    Mean value of every interest column + demographic breakdown, per cluster.
    This is the "demographic profiling" deliverable: what does a typical
    member of each segment look like?
    """
    profile = df_with_clusters.groupby(cluster_col)[config.INTEREST_COLS].mean()
    demo = df_with_clusters.groupby(cluster_col).agg(
        avg_age=("age", "mean"),
        avg_friends=("NumberOffriends", "mean"),
        n_students=(cluster_col, "size"),
    )
    gender_pct = (df_with_clusters.groupby(cluster_col)["gender"]
                  .value_counts(normalize=True).unstack().fillna(0) * 100).round(1)
    return profile, demo, gender_pct


def plot_cluster_theme_heatmap(df_with_clusters: pd.DataFrame, cluster_col: str = "cluster",
                                theme_map=config.INTEREST_THEMES):
    theme_means = {}
    for theme, cols in theme_map.items():
        theme_means[theme] = df_with_clusters.groupby(cluster_col)[cols].mean().mean(axis=1)
    theme_df = pd.DataFrame(theme_means)

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(theme_df) + 2))
    sns.heatmap(theme_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Average theme intensity per cluster")
    ax.set_xlabel("Interest theme")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return fig


def plot_cluster_trend_over_years(df_with_clusters: pd.DataFrame, cluster_col: str = "cluster"):
    """Cluster composition (%) by graduation year -- trend analysis over time."""
    ct = pd.crosstab(df_with_clusters["gradyear"], df_with_clusters[cluster_col],
                      normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_ylabel("% of students")
    ax.set_xlabel("Graduation year")
    ax.set_title("Cluster composition by graduation year")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig
