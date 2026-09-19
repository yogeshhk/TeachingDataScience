"""
Session 19 (MLOps): Step 5 of the "Deploy Your Own Model" section: a data-drift check.

Compare the weights the model was trained on with a batch of new requests using a
Kolmogorov-Smirnov test. Two simulated batches:
  new_ok    similar to the training data      -> p-value is large (no drift)
  new_heavy 20% heavier cars                  -> p-value is tiny (drift: investigate, maybe retrain)

A drift alert is a reason to INVESTIGATE, not an automatic command to retrain.

Data: datasets/shared/cars.csv

Run: conda activate mlcoep && python drift_check.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "shared", "cars.csv",
)

cars = pd.read_csv(DATA_PATH, sep=';')
cols = ['EngineSize', 'Horsepower', 'Weight']
cars[cols + ['MPG_City']] = cars[cols + ['MPG_City']].apply(pd.to_numeric, errors='coerce')
cars = cars.dropna(subset=cols + ['MPG_City'])

# train_weight: weights in the training data
train_weight = cars['Weight'].values

rng = np.random.default_rng(0)
new_ok = rng.choice(train_weight, 200) + rng.normal(0, 50, 200)
new_heavy = new_ok * 1.2      # a batch with 20% heavier cars

print(ks_2samp(train_weight, new_ok).pvalue.round(3))
print(ks_2samp(train_weight, new_heavy).pvalue < 0.01)
