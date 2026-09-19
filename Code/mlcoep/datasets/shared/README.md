# Shared Datasets (Sessions 16, 17, 19)

**Purpose**: One dataset used by three sessions, kept in one place so the copies cannot drift apart.
The cars case studies in `LaTeX/ml_kmeans_cars_case_study.tex` (Session 16), `LaTeX/ml_pca_cars_case_study.tex`
(Session 17) and the deploy-your-own-model walkthrough in `LaTeX/ml_mlops_workflow.tex` (Session 19) all read it.

## Files

| File | Rows | Columns | Used for |
|---|---|---|---|
| `cars.csv` | 428 | 16 | K-Means on engine specs (Session 16), PCA on the same specs (Session 17), predicting city mileage `MPG_City` with a random forest (Session 19) |

Columns: `Obs`, `Make`, `Model`, `Type`, `Origin`, `DriveTrain`, `MSRP`, `Invoice`, `EngineSize`, `Cylinders`,
`Horsepower`, `MPG_City`, `MPG_Highway`, `Weight`, `Wheelbase`, `Length`.

Two things to know before reading it:

- The separator is a **semicolon**, not a comma.
- `MSRP` and `Invoice` are text such as `$36,945`. Two rotary-engine cars have `.` in `Cylinders`, so the column
  loads as text; the scripts convert it with `pd.to_numeric(..., errors='coerce')` and drop those 2 rows (428 to 426 cars).

## Usage

```python
import pandas as pd

cars = pd.read_csv('cars.csv', sep=';')
cars['Cylinders'] = pd.to_numeric(cars['Cylinders'], errors='coerce')
cars = cars.dropna(subset=['EngineSize', 'Cylinders', 'Horsepower', 'MPG_City', 'Weight'])
```

Runnable scripts: `sessions/session16_kmeans/kmeans_cars_case_study.py`, `sessions/session17_pca/pca_cars_case_study.py`,
`sessions/session19_mlops/train_model.py`.

## Verification

Verified against `pandas 2.2.3` / `scikit-learn 1.7.2`: after dropping the 2 rows, 426 cars remain. With `k=3`,
standardizing first moves 107 of 426 cars (25%) to a different group than clustering the raw features, which is the
number the Session 16 slides quote. The silhouette score for `k = 3` is 0.457 (best of `k = 2..6`).

## Provenance

The file is byte-identical to `Code/ml/data/cars.csv` in this repository. It is the SAS sample data set `SASHELP.CARS`
(428 cars, 15 variables, plus an `Obs` counter column), exported with a semicolon separator and formatted prices. This was
checked on 2026-09-19: the first rows match the copy published in the `sassoftware/sas-viya-programming` repository value for
value (for example the Acura MDX: MSRP 36,945, invoice 33,337, engine 3.5, 265 hp, weight 4,451). The slides for Sessions 16
and 17 now name the data set. Where SAS itself obtained the underlying car specifications was not checked.
