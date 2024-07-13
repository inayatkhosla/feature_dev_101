"""
Utility functions for clustering
"""
from typing import List

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def scale_standard(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the input DataFrame using the StandardScaler.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the features to be scaled.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with the scaled features.

    Example:
    --------
    scaled_data = scale_standard(data)

    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def reduce_tsne(
    data: pd.DataFrame, n_components: int, random_seed: int
) -> pd.DataFrame:
    """
    Reduce the dimensionality of the input DataFrame using t-SNE.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data to be dimensionality reduced.
    n_components : int
        The number of components or dimensions in the reduced space. Typically 2 or 3.
    random_seed : int
        The seed used by the random number generator for reproducibility.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with the data transformed into the lower-dimensional space
        using t-SNE

    Example:
    --------
    reduced_data = reduce_tsne(data, n_components=2, random_seed=42)

    """
    tsne = TSNE(n_components=n_components, random_state=random_seed)
    X_tsne = tsne.fit_transform(data)
    clustered_data_tsne = pd.DataFrame(
        X_tsne, columns=[f"t-SNE{str(i)}" for i in range(1, n_components + 1)]
    )
    return clustered_data_tsne


def cluster_KMeans(
    data_scaled: pd.DataFrame, num_clusters: int, random_seed: int
) -> List[int]:
    """
    Cluster scaled data using the KMeans algorithm.

    Parameters:
    ----------
    data_scaled : pd.DataFrame
        The scaled input DataFrame containing the data to be clustered.
    num_clusters : int
        The number of clusters (k) to create.
    random_seed : int
        The seed used by the random number generator for reproducibility.

    Returns:
    -------
    List[int]
        A list of cluster labels assigned to each data point in the input DataFrame.

    Example:
    --------
    cluster_labels = cluster_KMeans(scaled_data, num_clusters=2, random_seed=42)

    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed)
    kmeans.fit(data_scaled)
    return kmeans.labels_
