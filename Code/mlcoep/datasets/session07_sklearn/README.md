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

Verified against `pandas 2.2.3` / `scikit-learn 1.7.2`: running the deck's code against these local
files reproduces every printed result on the slides exactly, including
`Accuracy: 0.772 (0.050)` and `R^2: 0.718 (0.099)`.

Note that the `R^2` figure depends on `shuffle=True` in the `KFold` split. The housing data is
stored in sorted order, so unshuffled folds are unrepresentative and the score drops to
`0.203 (0.595)`, with one fold going negative. This is called out on the slide.

## Provenance

Both files are mirrors of the public `jbrownlee/Datasets` repository, which is the source the
slides already cite (Jason Brownlee, Machine Learning Mastery).
