"""
Session 9 (Linear Regression): generates the synthetic housing_data.csv used by the last frame of
LaTeX/ml_linearregression_housing.tex.

The deck's story: a house price depends on square footage, bedrooms and neighborhood, and the
price per square foot differs by neighborhood (100 / 200 / 400). The data are simulated with a
fixed seed, so every student gets the same file.

  neighborhood_encoded: 0 = "skid row", 1 = average, 2 = "hipsterton"

Run: conda activate mlcoep && python make_housing_data.py
"""
import os

import numpy as np
import pandas as pd

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets",
    "session09_regression", "housing_data.csv",
)

rng = np.random.RandomState(42)
n = 500

neighborhood = rng.choice([0, 1, 2], size=n, p=[0.25, 0.5, 0.25])
sqft = rng.uniform(600, 3500, size=n).round()
bedrooms = np.clip(np.round(sqft / 900 + rng.normal(0, 0.7, size=n)), 1, 6).astype(int)

price_per_sqft = np.array([100, 200, 400])[neighborhood]
price = price_per_sqft * sqft + bedrooms * 1000 + rng.normal(0, 15000, size=n)

df = pd.DataFrame({
    "bedrooms": bedrooms,
    "sqft": sqft.astype(int),
    "neighborhood_encoded": neighborhood,
    "price": price.round().astype(int),
})
df.to_csv(OUT_PATH, index=False)
print("Wrote", os.path.normpath(OUT_PATH), df.shape)
print(df.head())
