"""
Session 15 (K-Nearest Neighbors): the NBA case study from LaTeX/ml_knn_nba_case_study.tex.

  1. Load nba_2013.csv and look at the columns.
  2. Distance of every player to LeBron James on RAW stats (large-scale columns dominate).
  3. Normalize the columns first, then find the most similar player.
  4. Split the players into train and test sets and predict points ("pts") with
     KNeighborsRegressor; report the mean squared error.

Data: datasets/session15_knn/nba_2013.csv

Note: like the slide, the test set is taken from random_indices[1:test_cutoff] (it skips the
first shuffled row). That is harmless here. A seed makes the split repeatable.

Run: conda activate mlcoep && python nba_similar_players.py
"""
import math
import os

import numpy as np
import pandas
from numpy.random import permutation
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets",
    "session15_knn", "nba_2013.csv",
)

# ---- block 1: load ---------------------------------------------------------
with open(DATA_PATH, 'r') as csvfile:
    nba = pandas.read_csv(csvfile)

print("Rows, columns:", nba.shape)
print(nba.columns.values)

# ---- block 2: raw Euclidean distance to LeBron -----------------------------
selected_player = nba[nba["player"] == "LeBron James"].iloc[0]

distance_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga',
                    'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p',
                    'x2pa', 'x2p.', 'efg.', 'ft', 'fta',
                    'ft.', 'orb', 'drb', 'trb', 'ast', 'stl',
                    'blk', 'tov', 'pf', 'pts']


def euclidean_distance(row):
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_player[k]) ** 2
    return math.sqrt(inner_value)


lebron_distance = nba.apply(euclidean_distance, axis=1)
raw_closest = nba.loc[lebron_distance.sort_values().index[1], "player"]
print("\nClosest to LeBron on RAW stats:", raw_closest)

# ---- blocks 3-4: normalize, then distance ----------------------------------
nba_numeric = nba[distance_columns]
nba_normalized = (nba_numeric - nba_numeric.mean()) / nba_numeric.std()
nba_normalized.fillna(0, inplace=True)

# .iloc[0] gives a 1-D row: current SciPy rejects the 2-D one-row DataFrame that the slide passes
lebron_normalized = nba_normalized[nba["player"] == "LeBron James"].iloc[0]
euclidean_distances = nba_normalized.apply(
    lambda row: distance.euclidean(row, lebron_normalized), axis=1)

distance_frame = pandas.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort_values("dist", inplace=True)
second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_lebron = nba.loc[int(second_smallest)]["player"]
print("Most similar to LeBron after normalizing:", most_similar_to_lebron)

# ---- blocks 5-7: KNN regression of points ----------------------------------
np.random.seed(1)
random_indices = permutation(nba.index)
test_cutoff = math.floor(len(nba) / 3)
test = nba.loc[random_indices[1:test_cutoff]]
train = nba.loc[random_indices[test_cutoff:]]

x_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.',
             'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']
y_column = ["pts"]

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(train[x_columns].fillna(0), train[y_column])
predictions = knn.predict(test[x_columns].fillna(0))

actual = test[y_column]
mse = (((predictions - actual) ** 2).sum()) / len(predictions)
print("\nTrain rows: %d, test rows: %d" % (len(train), len(test)))
print("Mean squared error of the points prediction:", float(mse.iloc[0]) if hasattr(mse, 'iloc') else float(mse))
