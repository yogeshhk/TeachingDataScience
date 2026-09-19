"""
Session 9 (Linear Regression): the scikit-learn house-price model from the last frame of
LaTeX/ml_linearregression_housing.tex, plus one extension that is NOT on the slides.

Data: datasets/session09_regression/housing_data.csv (synthetic; created by make_housing_data.py).

Run: conda activate mlcoep && python house_price_regression.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets",
    "session09_regression", "housing_data.csv",
)

# ---- as on the slide -------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X = df[['bedrooms', 'sqft', 'neighborhood_encoded']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R^2 Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# ---- extension (not on the slides) ----------------------------------------
# The price PER SQUARE FOOT changes with the neighborhood (100 / 200 / 400), so sqft and the
# neighborhood interact. A straight-line model cannot see that unless we add an interaction column.
df['sqft_x_neighborhood'] = df['sqft'] * df['neighborhood_encoded']
X2 = df[['bedrooms', 'sqft', 'neighborhood_encoded', 'sqft_x_neighborhood']]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42)
model2 = LinearRegression().fit(X2_train, y2_train)
print(f"\nWith the sqft x neighborhood interaction: R^2 = "
      f"{r2_score(y2_test, model2.predict(X2_test)):.3f}")
