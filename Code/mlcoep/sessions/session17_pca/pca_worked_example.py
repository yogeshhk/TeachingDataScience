"""
Session 17 (PCA): the ten-point PCA example worked by hand in LaTeX/ml_pca.tex
("Worked Example: PCA by Hand"), reproduced with NumPy.

  subtract the mean -> covariance matrix -> eigenvectors and eigenvalues -> keep the top
  component -> explained variance ratio -> project.

The numbers match the slides: covariance [[0.6166, 0.6154], [0.6154, 0.7166]],
eigenvalues 0.0491 and 1.2840, so the first component keeps 96.3% of the variance.

Run: conda activate mlcoep && python pca_worked_example.py
"""
import numpy as np

x = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1])
y = np.array([2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9])

# 1. subtract the mean of each feature
X = np.c_[x - x.mean(), y - y.mean()]
print("means:", x.mean(), y.mean())

# 2. covariance matrix (divide by n - 1)
C = np.cov(X.T)
print("covariance matrix:")
print(C.round(4))

# 3. eigenvalues and eigenvectors (eigh: for symmetric matrices, ascending order)
eigenvalues, eigenvectors = np.linalg.eigh(C)
print("eigenvalues:", eigenvalues.round(4))
print("eigenvectors (columns):")
print(eigenvectors.round(4))
print("perpendicular:", bool(abs(eigenvectors[:, 0] @ eigenvectors[:, 1]) < 1e-12))

# 4. order by eigenvalue, largest first
order = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

# 5. explained variance ratio
ratio = eigenvalues / eigenvalues.sum()
print("explained variance ratio: %.1f%% and %.1f%%" % (100 * ratio[0], 100 * ratio[1]))

# 6. project onto the first component: one number per point
W = eigenvectors[:, :1]
Z = X @ W
print("projected data (first 3 points):", Z[:3].ravel().round(3))
