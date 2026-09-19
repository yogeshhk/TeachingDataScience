"""
Session 15 (K-Nearest Neighbors): the from-scratch KNN of LaTeX/ml_knn.tex (train / predict /
kNearestNeighbor / accuracy), run on the Iris data set that ships with scikit-learn.

The slides leave X_train, y_train, X_test, y_test undefined; Iris with a 1/3 test split is a
convenient stand-in. At the end we compare with scikit-learn's KNeighborsClassifier.

Run: conda activate mlcoep && python knn_from_scratch.py
"""
from collections import Counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train(X_train, y_train):
    # do nothing: KNN is a "lazy" learner
    return


def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        distances.append([distance, i])

    distances = sorted(distances)

    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    return Counter(targets).most_common(1)[0][0]


def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
    train(X_train, y_train)

    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.33, random_state=42)

# making our predictions
predictions = []
kNearestNeighbor(X_train, y_train, X_test, predictions, 7)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
print('\nThe accuracy is %d%%' % (accuracy * 100))

# the same thing with scikit-learn
sk = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
print('scikit-learn KNeighborsClassifier(7) accuracy: %d%%' % (accuracy_score(y_test, sk.predict(X_test)) * 100))
print('Same predictions as the from-scratch version:', bool(np.array_equal(sk.predict(X_test), predictions)))
