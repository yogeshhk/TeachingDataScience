"""
Session 18 (Titanic Capstone): the end-to-end pipeline of LaTeX/ml_titanic_sklearn.tex, in one
script: load -> explore -> clean and engineer features -> random forest -> held-out evaluation ->
cross-validation -> feature importances.

Data: datasets/session18_titanic/titanic_train.csv and titanic_test.csv (Kaggle Titanic).

Run: conda activate mlcoep && python titanic_random_forest.py
"""
import os

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "session18_titanic",
)

df_test = pd.read_csv(os.path.join(DATA_DIR, "titanic_test.csv"))
df_train = pd.read_csv(os.path.join(DATA_DIR, "titanic_train.csv"))

print("Dimensions of train: {}".format(df_train.shape))
print("Dimensions of test: {}".format(df_test.shape))
print("\nMissing values in train:", df_train.isnull().sum()[lambda s: s > 0].to_dict())
print("Survival by sex:", df_train.groupby('Sex')['Survived'].mean().round(3).to_dict())
print("Survival by class:", df_train.groupby('Pclass')['Survived'].mean().round(3).to_dict())


def clean_data(df, drop_passenger_id):
    sexes = sorted(df['Sex'].unique())
    df['Sex_Val'] = df['Sex'].map(dict(zip(sexes, range(len(sexes))))).astype(int)
    df['AgeFill'] = df.groupby(['Sex_Val', 'Pclass'])['Age'] \
                      .transform(lambda x: x.fillna(x.median()))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked',
                  'Age', 'SibSp', 'Parch'], axis=1)
    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)
    return df


df_train = clean_data(df_train, drop_passenger_id=True)
train_data = df_train.values
print("\nColumns after cleaning:", list(df_train.columns))

# Training data features, skip the first column 'Survived'
train_features = train_data[:, 1:]
# 'Survived' column values
train_target = train_data[:, 0]

# ---- Random forest: fit and score on the training data ---------------------
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
print("\nMean accuracy of Random Forest on the TRAINING data: {0}".format(score))

# ---- Held-out evaluation ---------------------------------------------------
train_x, test_x, train_y, test_y = train_test_split(
    train_features, train_target, test_size=0.20, random_state=0)
print("Shapes:", train_features.shape, train_x.shape, test_x.shape)

clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
print("Accuracy on held-out data = %.2f" % accuracy_score(test_y, predict_y))
print("\nConfusion matrix:")
print(metrics.confusion_matrix(test_y, predict_y))
print(classification_report(test_y, predict_y, target_names=['Not Survived', 'Survived']))

# ---- Cross-validation ------------------------------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=0)
scores = cross_val_score(clf, train_features, train_target, cv=5)
print("5-fold scores:", scores.round(3))
print("Accuracy: %.3f +/- %.3f" % (scores.mean(), scores.std()))

# ---- Feature importances ---------------------------------------------------
clf = clf.fit(train_features, train_target)
names = ['Pclass', 'Fare', 'Sex_Val', 'AgeFill', 'FamilySize']
print("\nFeature importances:")
for name, value in zip(names, clf.feature_importances_):
    print("%-10s %.3f" % (name, value))

# ---- The test set gets the same cleaning -----------------------------------
df_test = clean_data(df_test, drop_passenger_id=False)
test_y = clf.predict(df_test.values[:, 1:])
print("\nPredicted survivors in the test set: %d of %d" % (int(test_y.sum()), len(test_y)))
