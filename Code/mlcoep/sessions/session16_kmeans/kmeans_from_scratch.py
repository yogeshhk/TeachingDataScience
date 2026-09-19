"""
Session 16 (K-Means): the from-scratch K-Means of LaTeX/ml_kmeans.tex ("K-Means from Scratch"
frames), on the ex7data2.mat points.

  find_closest_centroids -> compute_centroids -> run_k_means, plus random initialization.

The code and the printed numbers follow Andrew Ng's Coursera Machine Learning exercise 7
(Python port by John Wittenauer), as credited on the slide.

Data: datasets/session16_kmeans/ex7data2.mat

Run: conda activate mlcoep && python kmeans_from_scratch.py
"""
import os

import numpy as np
from scipy.io import loadmat

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets",
    "session16_kmeans", "ex7data2.mat",
)

data = loadmat(DATA_PATH)
X = data['X']
print("Data shape:", X.shape)


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j

    return idx


def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) /
                           len(indices[0])).ravel()
    return centroids


def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    # distinct points: no two centroids coincide
    idx = np.random.choice(m, k, replace=False)
    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids


def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    # fixed number of rounds; production code also stops early once centroids stop moving
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids


initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = find_closest_centroids(X, initial_centroids)
print("idx[0:3]:", idx[0:3])
print("compute_centroids after one assignment:")
print(compute_centroids(X, idx, 3))

np.random.seed(0)
print("init_centroids (random, seed 0):")
print(init_centroids(X, 3))

idx, centroids = run_k_means(X, initial_centroids, 10)
print("\nFinal centroids after 10 rounds:")
print(centroids.round(3))
print("Cluster sizes:", np.bincount(idx.astype(int)))
