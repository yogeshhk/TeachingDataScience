"""
Session 20 (ME Applications): K-Means customer segmentation, from LaTeX/ml_course_demo_clustering_customers.tex.

  1. Simulate 3 customer segments (budget, standard, premium) with a fixed seed.
  2. Standardize, then run the elbow method for K = 1..9.
  3. Fit K-Means with K = 3, name the clusters by sorting on mean income, print the segment profiles.

Results the slides quote:
  inertia for K = 1..9: 720.0  216.9  87.7  76.5  65.2  54.8  48.2  41.4  37.3
  segment means (income, spending): Budget 29.3 / 20.7, Standard 55.4 / 54.7, Premium 79.0 / 84.6

Data: synthetic, generated in the script (np.random.seed(42)); no download.

Run: conda activate mlcoep && python demo_customer_segments.py
(On a machine without a display the figures are saved as PNG files next to this script.)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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


# ---- Generate the data -------------------------------------------------------
np.random.seed(42)
# Simulate 3 customer segments: budget, standard, premium
centers = [(30, 20), (55, 55), (80, 85)]
X = np.vstack([
    np.random.randn(120, 2) * [8, 10] + c
    for c in centers])
df = pd.DataFrame(X, columns=['AnnualIncome', 'SpendingScore'])
print(df.describe())

# ---- Elbow method ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

inertia = []
K_range = range(1, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

print("\nInertia for K = 1..9:", "  ".join("%.1f" % v for v in inertia))

plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters K")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.grid(True)
finish("customers_elbow.png")

# ---- Fit K-Means with K = 3 --------------------------------------------------
km = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Segment'] = km.fit_predict(X_scaled)

# KMeans cluster numbers are arbitrary, so map them to
# Budget/Standard/Premium by sorting on mean income
order = df.groupby('Segment')['AnnualIncome'].mean().sort_values().index
labels = dict(zip(order, ['Budget', 'Standard', 'Premium']))
colors = {'Budget': '#e07b54', 'Standard': '#5b9bd5', 'Premium': '#70ad47'}

plt.figure(figsize=(7, 5))
for seg, name in labels.items():
    mask = df['Segment'] == seg
    plt.scatter(df.loc[mask, 'AnnualIncome'],
                df.loc[mask, 'SpendingScore'],
                label=name, color=colors[name], alpha=0.7)
plt.xlabel("Annual Income (k USD)")
plt.ylabel("Spending Score")
plt.title("Customer Segments (K-Means, K=3)")
plt.legend()
plt.tight_layout()
finish("customers_segments.png")

# ---- Segment profiles --------------------------------------------------------
profile = df.groupby('Segment')[
    ['AnnualIncome', 'SpendingScore']].mean()
profile.index = [labels[i] for i in profile.index]
print("\n" + str(profile.sort_values('AnnualIncome').round(1)))
