"""
Session 19 (MLOps): Step 1 of the "Deploy Your Own Model" section (LaTeX/ml_mlops_workflow.tex):
train a city-MPG model on the cars data and SAVE it with joblib.

The service (app.py) only needs the saved file, mpg_model.joblib, which is created next to
this script. It is not committed to git (see .gitignore): everyone builds it locally.

Features: engine size, horsepower, weight.  Target: city MPG.
Data: datasets/shared/cars.csv

Run: conda activate mlcoep && python train_model.py
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "..", "datasets", "shared", "cars.csv")
MODEL_PATH = os.path.join(HERE, "mpg_model.joblib")

cars = pd.read_csv(DATA_PATH, sep=';')
cols = ['EngineSize', 'Horsepower', 'Weight']
cars[cols + ['MPG_City']] = cars[cols + ['MPG_City']].apply(pd.to_numeric, errors='coerce')
cars = cars.dropna(subset=cols + ['MPG_City'])

# X: engine size, horsepower, weight of each car; y: city MPG
X, y = cars[cols].values, cars['MPG_City'].values

model = RandomForestRegressor(n_estimators=100, random_state=0).fit(X, y)
joblib.dump(model, MODEL_PATH)        # save
print("Trained on %d cars, saved %s (%d KB)" % (len(cars), os.path.basename(MODEL_PATH), os.path.getsize(MODEL_PATH) // 1024))

# In a new session: load the file and predict
loaded = joblib.load(MODEL_PATH)
print("predict [[2.0, 140, 2800]] ->", loaded.predict([[2.0, 140, 2800]]).round(1))
