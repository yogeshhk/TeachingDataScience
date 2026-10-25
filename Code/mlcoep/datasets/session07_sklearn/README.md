# Sklearn Workflow Datasets (Session 7)

**Purpose**: Offline copies of the two datasets used in the Session 7 slides
(`LaTeX/ml_datapreparation_sklearn.tex`, `LaTeX/ml_evaluation_sklearn.tex`), so the session runs
without internet access.

Both decks load these over HTTPS from the `jbrownlee/Datasets` GitHub repo. If the classroom
network is down or the upstream repo moves, point `read_csv` at the local file instead.

## Files

| File | Rows | Columns | Used for |
|---|---|---|---|
| `pima-indians-diabetes.data.csv` | 768 | 9 | Preprocessing (rescale, standardize, normalize, binarize) and all classification metrics |
| `housing.data` | 506 | 14 | Regression metrics (MAE, MSE, R squared) |

Neither file has a header row; the decks supply column names explicitly via the `names=` argument.

## Usage

Replace the URL in the deck's code with the local path:

```python
# instead of the https://raw.githubusercontent.com/... URL
dataframe = pandas.read_csv('pima-indians-diabetes.data.csv', names=names)

# housing.data is whitespace-separated, not comma-separated
dataframe = pandas.read_csv('housing.data', sep='\s+', names=names)
```

## Verification

Run the deck's code on these local files and compare with the printed slide results; the housing
`R^2` needs `shuffle=True` in `KFold` because the file is stored in sorted order.

## Provenance

Both files are mirrors of the public `jbrownlee/Datasets` repository, which is the source the
slides already cite (Jason Brownlee, Machine Learning Mastery).
