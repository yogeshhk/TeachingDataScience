# Shared Datasets

**Purpose**: Datasets used by more than one session, kept in one place so the copies cannot drift apart.

## Files

| File | Rows | Columns | Separator | Description |
|---|---|---|---|---|
| `cars.csv` | 428 | 16 | semicolon | Car specifications and prices (SAS sample data `SASHELP.CARS`) |
| `Crop_recommendation.csv` | 2200 | 8 | comma | Soil and climate measurements labelled with the best-suited crop |

## Data dictionary

### `cars.csv`

`Obs`, `Make`, `Model`, `Type`, `Origin`, `DriveTrain`, `MSRP`, `Invoice`, `EngineSize`, `Cylinders`, `Horsepower`,
`MPG_City`, `MPG_Highway`, `Weight`, `Wheelbase`, `Length`.

- `MSRP` and `Invoice` are text such as `$36,945`.
- Two rotary-engine cars have `.` in `Cylinders`, so the column loads as text.

### `Crop_recommendation.csv`

| Column | Meaning |
|---|---|
| `N`, `P`, `K` | Soil nitrogen, phosphorus and potassium content |
| `temperature` | Temperature in degrees Celsius |
| `humidity` | Relative humidity in percent |
| `ph` | Soil pH |
| `rainfall` | Rainfall in mm |
| `label` | Crop name (target, 22 classes, 100 rows each) |

There are no missing values, and all features are numeric.

## Reading the files

Check the separator and the column types first; they differ between files.

```python
import pandas as pd

cars = pd.read_csv('cars.csv', sep=';')
crops = pd.read_csv('Crop_recommendation.csv')

for df in (cars, crops):
    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum())
```

Generic cleaning steps, as needed:

- Convert text numbers with `pd.to_numeric(col, errors='coerce')`, then drop the rows that became `NaN`.
- Strip `$` and `,` from price text before converting.
- Standardize features (for example `StandardScaler`) before distance-based methods such as K-Means or PCA.
- For a supervised task, split into train and test sets with `train_test_split`, using `stratify` for a class label.

Paths in the scripts are relative to the file that reads them, so adjust them to where you run from.

## Provenance

- `cars.csv` is byte-identical to `Code/ml/data/cars.csv` in this repository. It is the SAS sample data set
  `SASHELP.CARS` (428 cars, 15 variables, plus an `Obs` counter column), exported with a semicolon separator and
  formatted prices. The first rows match the copy published in the `sassoftware/sas-viya-programming` repository.
  Where SAS itself obtained the underlying car specifications is not known.
- `Crop_recommendation.csv` is the "Crop Recommendation" data set (2200 rows, 22 crops), downloaded from Kaggle.
  The exact Kaggle page and its license are not recorded here; confirm the license there before redistributing
  it outside this repository.
