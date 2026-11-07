"""Clustering estimator factories with explicit linkage configuration."""
from sklearn.cluster import KMeans, AgglomerativeClustering


def create_kmeans(n_clusters=3, random_state=42, **kwargs):
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto", **kwargs)


def create_hac(n_clusters=3, linkage="ward", metric="euclidean", **kwargs):
    """Create hierarchical agglomerative clustering with explicit settings.

    Ward linkage only supports Euclidean distance.
    """
    if linkage == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage requires metric='euclidean'.")
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric, **kwargs)
