"""
Session 20 (ME Applications): House price prediction, from LaTeX/ml_course_demo_regression_housing.tex.

  1. Load the California Housing data (8 features, target = median house value in 100,000s USD).
  2. Split 80/20, standardize, fit Linear Regression, report RMSE and R2.
  3. Compare Linear, Ridge (alpha=1.0) and Lasso (alpha=0.1).
  4. Plot predicted against actual values.

Data: sklearn.datasets.fetch_california_housing downloads the file on first use (internet needed
once), then caches it under ~/scikit_learn_data.

Results the slides quote:
  Linear    RMSE=0.7456  R2=0.5758
  Ridge     RMSE=0.7456  R2=0.5758
  Lasso     RMSE=0.8244  R2=0.4814

Run: conda activate mlcoep && python demo_housing_regression.py
(On a machine without a display the figure is saved as housing_pred_vs_actual.png next to this script.)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Load and explore -------------------------------------------------------
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print(df.shape)       # (20640, 9)
print(df.describe())
print(df.head())

# ---- Train Linear Regression -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2  :", r2_score(y_test, y_pred))

# ---- Compare regularization --------------------------------------------------
print()
for name, model in [("Linear", LinearRegression()),
                    ("Ridge", Ridge(alpha=1.0)),
                    ("Lasso", Lasso(alpha=0.1))]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name:8s}  RMSE={rmse:.4f}  R2={r2:.4f}")

# ---- Predicted vs actual (linear model) --------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Predicted vs Actual")
plt.tight_layout()

if plt.get_backend().lower() == 'agg':
    out = os.path.join(os.path.dirname(__file__), "housing_pred_vs_actual.png")
    plt.savefig(out)
    print("Saved", out)
else:
    plt.show()
