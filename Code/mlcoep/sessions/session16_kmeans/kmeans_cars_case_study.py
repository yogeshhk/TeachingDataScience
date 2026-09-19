"""
Session 16 (K-Means): the cars case study from LaTeX/ml_kmeans_cars_case_study.tex.

  1. Load and clean cars.csv (2 rotary-engine cars have '.' cylinders and are dropped).
  2. K-Means (k=3) on the RAW features: Weight (in the thousands) decides the clusters.
  3. Standardize first, then K-Means: 25% of the cars change group.
  4. Is k=3 right? Inertia (elbow) and silhouette score for k = 2..6.

Data: datasets/shared/cars.csv  (also used by Sessions 17 and 19)

Run: conda activate mlcoep && python kmeans_cars_case_study.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "shared", "cars.csv",
)

# ---- The Data ---------------------------------------------------------------
cars = pd.read_csv(DATA_PATH, sep=';')
cars['Cylinders'] = pd.to_numeric(cars['Cylinders'],
                                  errors='coerce')  # '.' becomes NaN

features = ['EngineSize', 'Cylinders', 'Horsepower',
            'MPG_City', 'Weight']
cars = cars.dropna(subset=features)     # 428 -> 426 cars
X = cars[features]
print("Cars used:", len(cars))

# ---- Cluster the Raw Features ----------------------------------------------
km = KMeans(n_clusters=3, n_init=10, random_state=0)
km.fit(X)
cars['cluster'] = km.labels_
raw_labels = km.labels_.copy()

print("\nRaw features: cluster averages")
print(cars.groupby('cluster')[features].mean().round(1).T)
print("Weight variance: %.0f, largest other variance: %.0f" % (
    X['Weight'].var(), X.drop(columns='Weight').var().max()))

# ---- Scale First -----------------------------------------------------------
Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, n_init=10, random_state=0)
km.fit(Xs)
cars['cluster'] = km.labels_

print("\nStandardized features: cluster averages")
print(cars.groupby('cluster')[features].mean().round(1).T)

# how many cars change group? (match the two labelings with the best permutation)
from scipy.optimize import linear_sum_assignment
C = pd.crosstab(raw_labels, km.labels_).values
r, c = linear_sum_assignment(-C)
same = C[r, c].sum()
print("\nCars in a different group than with raw features: %d of %d (%.0f%%)" % (
    len(cars) - same, len(cars), 100.0 * (len(cars) - same) / len(cars)))

# ---- Is k=3 Right? ---------------------------------------------------------
print("\nk  inertia  silhouette")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(Xs)
    sil = silhouette_score(Xs, km.labels_)
    print(k, round(km.inertia_, 1), round(sil, 3))
