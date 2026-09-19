"""
Session 20 (ME Applications): PCA + K-Means on handwritten digits, from LaTeX/ml_course_assign_pca_digits.tex.

This is an assignment. The script runs Parts A, B and C exactly as the slides show them, then adds
a starter for the last three tasks (scree plot, best configuration, cluster centers) so students can
compare their own answers.

  Part A: standardize the 64 pixel features, project to 2 PCA components, plot by digit.
  Part B/C: K-Means (K=10) on 2, 10 and 30 PCA components; silhouette score and Adjusted Rand Index.
  Tasks: variance explained by 2, 10 and 30 components; scree plot; where ARI peaks; cluster centers
         mapped back to pixel space (inverse_transform).

Data: sklearn.datasets.load_digits (built in, no download).

Run: conda activate mlcoep && python assign_pca_digits.py
(On a machine without a display the figures are saved as PNG files next to this script.)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
HEADLESS = plt.get_backend().lower() == 'agg'


def finish(name):
    """Save the current figure on a headless machine, otherwise show it."""
    if HEADLESS:
        out = os.path.join(HERE, name)
        plt.savefig(out)
        print("Saved", out)
        plt.close()
    else:
        plt.show()


# ---- Part A: PCA visualization -----------------------------------------------
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

pca2 = PCA(n_components=2)
X2 = pca2.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X2[:, 0], X2[:, 1],
                      c=digits.target, cmap='tab10',
                      s=10, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title("Digits projected to 2 PCA components")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
finish("digits_pca2.png")

print("Variance explained:", pca2.explained_variance_ratio_.sum())

# ---- Parts B and C: K-Means with varying PCA components ----------------------
print()
results = {}
for n_comp in [2, 10, 30]:
    Xr = PCA(n_components=n_comp).fit_transform(X)
    km = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels = km.fit_predict(Xr)
    sil = silhouette_score(Xr, labels)
    ari = adjusted_rand_score(digits.target, labels)
    results[n_comp] = ari
    print(f"PCA={n_comp:2d}  Silhouette={sil:.3f}  "
          f"ARI={ari:.3f}")

# ---- Starter for the remaining tasks -----------------------------------------
print()
pca_full = PCA().fit(X)
cum = np.cumsum(pca_full.explained_variance_ratio_)
for n_comp in [2, 10, 30]:
    print(f"Variance explained by {n_comp:2d} components: {cum[n_comp - 1]:.3f}")

plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(cum) + 1), pca_full.explained_variance_ratio_, marker='o', ms=3)
plt.xlabel("Component")
plt.ylabel("Explained variance ratio")
plt.title("Scree plot (Digits, standardized)")
plt.grid(True)
plt.tight_layout()
finish("digits_scree.png")

best = max(results, key=results.get)
print(f"\nARI peaks at PCA={best} (ARI={results[best]:.3f}) among 2, 10, 30")

pca_best = PCA(n_components=best)
Xb = pca_best.fit_transform(X)
km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(Xb)
# cluster centers live in PCA space: map back to standardized pixels, then to the original scale
centers = StandardScaler().fit(digits.data).inverse_transform(pca_best.inverse_transform(km.cluster_centers_))

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, c, k in zip(axes.ravel(), centers, range(10)):
    ax.imshow(c.reshape(8, 8), cmap='gray_r')
    ax.set_title(f"Cluster {k}")
    ax.axis('off')
plt.suptitle(f"Cluster centers (K-Means on {best} PCA components)")
plt.tight_layout()
finish("digits_cluster_centers.png")
