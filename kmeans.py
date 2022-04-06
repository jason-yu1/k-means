from math import dist
from turtle import distance
import numpy as np
import sys


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2):
    if centroids == 'kmeans++':
        centroids = select_centroids(X, k)
    else:
        centroids = X[np.random.choice(X.shape[0], k, replace=False), :]
    cluster_change = True
    labels = np.zeros(X.shape[0], dtype=int)
    num_iter = 0
    new_centroids = np.copy(centroids)
    while cluster_change:
        for i, x in enumerate(X):
            distances = np.linalg.norm(x - centroids, axis=1)
            j = np.argmin(distances)
            labels[i] = j
        
        for j in range(k):
            new_centroids[j] = np.sum(X[labels == j], axis=0) / X[labels == j].shape[0]

        if (np.mean(np.linalg.norm(new_centroids - centroids, axis=1)) < tolerance) | (num_iter > max_iter):
            return centroids, labels
        else:
            num_iter += 1
            centroids = new_centroids.copy()

    return centroids, labels


def select_centroids(X, k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    centroids = np.zeros((k, X.shape[1]))
    centroids[0] = X[np.random.choice(X.shape[0], 1), :]
    for i in range(1, k):
        existing = centroids[:i]
        min_distances = []
        for x in X:
            distances = np.linalg.norm(x - existing, axis=1)
            min_distance = np.min(distances)
            min_distances.append(min_distance)

        centroids[i] = X[np.argmax(min_distances), :]
    return centroids

