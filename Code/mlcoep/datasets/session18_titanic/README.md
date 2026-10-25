# Titanic Passenger Data (Session 18)

**Purpose**: Offline copy of the Titanic data for the capstone in `LaTeX/ml_titanic_sklearn.tex`, so the session
runs without internet access or a Kaggle login.

## Files

| File | Rows | Columns | Used for |
|---|---|---|---|
| `titanic_train.csv` | 891 | 12 | Cleaning, feature engineering, training and evaluating the random forest |
| `titanic_test.csv` | 418 | 11 | Predicting survival for unseen passengers (no `Survived` column) |

Columns in the training file: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`,
`Cabin`, `Embarked`. The test file has the same columns without `Survived`.

`Age`, `Cabin` and `Embarked` have missing values; the slides fill `Age` (as `AgeFill`) and derive `Sex_Val` and
`FamilySize`.

## Usage

```python
import pandas as pd

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')
```

Runnable script: `sessions/session18_titanic/titanic_random_forest.py`.

## Verification

Check the shapes above, and that `titanic_random_forest.py` reproduces the hold-out accuracy and feature importances shown on the slides.

## Provenance

Byte-identical to the files in `Code/ml/data`. The layout is the standard Kaggle "Titanic: Machine Learning from
Disaster" competition data (`train.csv` and `test.csv`).
