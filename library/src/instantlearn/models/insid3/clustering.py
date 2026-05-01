# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Agglomerative clustering utilities for INSID3."""

import torch


def agglomerative_clustering(features: torch.Tensor, tau: float) -> torch.Tensor:
    """Perform agglomerative clustering on patch features using cosine distance.

    Iteratively merges the most similar clusters until no pair exceeds the threshold tau.
    Uses Ward-like linkage with cosine similarity.

    Args:
        features: Normalized patch features of shape (P, D).
        tau: Distance threshold for stopping merges. Higher = finer clusters.

    Returns:
        Cluster labels of shape (P,) with integer IDs starting from 0.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    features_np = features.detach().cpu().float().numpy()

    # Compute cosine distances
    distances = pdist(features_np, metric="cosine")

    # Agglomerative clustering with average linkage
    linkage_matrix = linkage(distances, method="average")

    # Cut at threshold tau (cosine distance threshold = 1 - similarity threshold)
    # Paper uses tau as similarity threshold, scipy uses distance
    cluster_labels = fcluster(linkage_matrix, t=(1.0 - tau), criterion="distance")

    # Convert to 0-indexed tensor
    return torch.tensor(cluster_labels - 1, dtype=torch.long, device=features.device)


def compute_cluster_prototypes(features: torch.Tensor, labels: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """Compute the mean prototype for each cluster.

    Args:
        features: Patch features of shape (P, D).
        labels: Cluster labels of shape (P,).
        num_clusters: Total number of clusters K.

    Returns:
        Cluster prototypes of shape (K, D), L2-normalized.
    """
    D = features.shape[1]
    prototypes = torch.zeros(num_clusters, D, device=features.device, dtype=features.dtype)

    for k in range(num_clusters):
        mask = labels == k
        if mask.any():
            prototypes[k] = features[mask].mean(dim=0)

    return torch.nn.functional.normalize(prototypes, p=2, dim=-1)
