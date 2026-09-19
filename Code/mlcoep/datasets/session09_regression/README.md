# House Price Dataset (Session 9)

**Purpose**: The synthetic housing data for the last frame of `LaTeX/ml_linearregression_housing.tex`, so every
student fits the model on the same file and gets the same numbers.

The deck's story: a house price depends on square footage, bedrooms and neighborhood, and the price per square foot
differs by neighborhood (about 100, 200 and 400).

## Files

| File | Rows | Columns | Used for |
|---|---|---|---|
| `housing_data.csv` | 500 | 4 | Multiple linear regression, then the sqft by neighborhood interaction |

Columns: `bedrooms`, `sqft`, `neighborhood_encoded` (0 = "skid row", 1 = average, 2 = "hipsterton"), `price`.

## Usage

The file is **generated**, not downloaded. To recreate it (identical every time, the seed is fixed):

```
python sessions/session09_linear_regression/make_housing_data.py
```

Then run `sessions/session09_linear_regression/house_price_regression.py`.

## Verification

Verified against `pandas 2.2.3` / `scikit-learn 1.7.2`: 500 rows, no missing values, neighborhood counts 130 / 239 / 131,
mean price per square foot 103 / 201 / 402. A plain linear model on the three columns gives `R^2 = 0.859`. It rises to
`0.951` after adding a `sqft * neighborhood` interaction term, because the price per square foot is different in each
neighborhood, which a single `sqft` coefficient cannot express.

## Provenance

Synthetic. Produced by `make_housing_data.py` with `np.random.RandomState(42)`; no real housing data is involved.
