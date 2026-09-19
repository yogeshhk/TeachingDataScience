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

Verified against `pandas 2.2.3` / `scikit-learn 1.7.2`: the shapes above hold. The random forest scores 0.84 accuracy on a
179-passenger hold-out (recall 0.71 for survivors), and `0.817 +/- 0.030` over 5-fold cross-validation. It predicts 149 of
the 418 test passengers as survivors. Feature importances: `Fare` 0.294, `Sex_Val` 0.275, `AgeFill` 0.267, `Pclass` 0.083,
`FamilySize` 0.081.

## Provenance

Byte-identical to the files in `Code/ml/data`. The layout is the standard Kaggle "Titanic: Machine Learning from
Disaster" competition data (`train.csv` and `test.csv`).
