"""
Session 17 (PCA): the cars case study from LaTeX/ml_pca_cars_case_study.tex.

  raw features -> PC1 is just Weight; standardize -> real structure; loadings; PCA from
  scratch with NumPy (eigh); the same result via SVD; randomized PCA; project and reconstruct.

Data: datasets/shared/cars.csv  (also used by Sessions 16 and 19)

Run: conda activate mlcoep && python pca_cars_case_study.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "shared", "cars.csv",
)

# ---- The Data ---------------------------------------------------------------
cars = pd.read_csv(DATA_PATH, sep=';')
features = ['EngineSize', 'Cylinders', 'Horsepower', 'MPG_City',
            'MPG_Highway', 'Weight', 'Wheelbase', 'Length']
cars[features] = cars[features].apply(pd.to_numeric, errors='coerce')
cars = cars.dropna(subset=features)     # 428 -> 426 cars
X = cars[features]
print("Cars used:", len(cars))

# ---- Raw features: PC1 is Weight -------------------------------------------
pca = PCA()
pca.fit(X)
print("\nRaw: explained variance ratio (first 3):", pca.explained_variance_ratio_[:3].round(3))
print("Raw: PC1 loadings:", pca.components_[0].round(3))

# ---- Standardize, then PCA -------------------------------------------------
Xs = StandardScaler().fit_transform(X)
pca = PCA()
pca.fit(Xs)
print("\nStandardized: explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("Cumulative:", pca.explained_variance_ratio_.cumsum().round(3))

# ---- Reading the components ------------------------------------------------
load = pd.DataFrame(pca.components_[:2].T, index=features,
                    columns=['PC1', 'PC2'])
print("\nLoadings:")
print(load.round(2))

# ---- PCA from scratch with NumPy -------------------------------------------
C = np.cov(Xs.T)                     # 8 x 8 covariance matrix
eigvals, eigvecs = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]    # largest eigenvalue first
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
print("\nFrom scratch: eigenvalues:", eigvals.round(3))
print("Match scikit-learn explained_variance_:", np.allclose(eigvals, pca.explained_variance_))

# ---- The same result via SVD -----------------------------------------------
U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
print("\nSVD: sigma^2/(n-1):", (s ** 2 / (len(Xs) - 1)).round(3))
print("SVD: first component equals scikit-learn's (up to sign):",
      bool(np.allclose(np.abs(Vt[0]), np.abs(pca.components_[0]))))

# ---- Randomized PCA --------------------------------------------------------
pca_rand = PCA(n_components=3, svd_solver='randomized', random_state=0)
pca_rand.fit(Xs)
print("\nRandomized PCA ratio:", pca_rand.explained_variance_ratio_.round(3))

# ---- Project and reconstruct -----------------------------------------------
W = eigvecs[:, :2]        # projection matrix: 8 x 2
Z = Xs @ W                # 2 numbers per car
Xr = Z @ W.T              # back to 8 features
print("\nZ shape:", Z.shape)
print("Reconstruction error (k=2):", ((Xs - Xr) ** 2).mean().round(3))
print("Variance dropped (1 - kept):", (1 - pca.explained_variance_ratio_[:2].sum()).round(3))
