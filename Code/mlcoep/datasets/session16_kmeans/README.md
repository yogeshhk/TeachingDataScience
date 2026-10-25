# K-Means Practice Points (Session 16)

**Purpose**: The two-dimensional point cloud for the "K-Means from Scratch" frames in `LaTeX/ml_kmeans.tex`, so the
walkthrough runs without internet access.

## Files

| File | Contents | Used for |
|---|---|---|
| `ex7data2.mat` | one array `X` of shape (300, 2) | Finding the closest centroid, computing centroids, random initialization, the main loop |

The 300 points form three visible blobs, which is why `K = 3` is the natural choice.

## Usage

```python
from scipy.io import loadmat

X = loadmat('ex7data2.mat')['X']      # shape (300, 2)
```

Runnable script: `sessions/session16_kmeans/kmeans_from_scratch.py`, which follows the slide frames step by step.

## Verification

Check that `X` has shape (300, 2) and that `kmeans_from_scratch.py` ends with three centroids and clusters of about 100 points each.

## Provenance

The file name follows the exercise 7 data file of the Coursera Machine Learning course (Andrew Ng), which the from-scratch
frames also follow. That credit is inferred from the file name and the code, not confirmed against a source.
