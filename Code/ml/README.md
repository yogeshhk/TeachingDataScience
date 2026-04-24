# Machine Learning Examples

A large collection (~160 notebooks) covering classical ML algorithms and data analysis patterns. Content drawn from multiple authors and courses.

## Setup
```bash
conda env create -f environment.yml
conda activate ml
```

## Content Organization

Files are named `ml_<topic>_<source>.ipynb`. Key contributor suffixes:
- `_SusanLi` — data exploration, Airbnb/ad-demand forecasting, visualization
- `_mlcourseai` — structured assignments (pandas → decision trees → regression → time series → Kaggle)
- `_TirthajyotiSarkar` — classification comparisons, loan/financial data
- `_Vanderplas` — scikit-learn patterns (from *Python Data Science Handbook*)

## Learning Path (recommended order)

1. **Foundations** — `ml_03_Numpy_Noteboo_SusanLi.ipynb`, `ml_05a_Matplotlib_Noteboo_SusanLi.ipynb`
2. **Classification** — `ml_assignment03_decision_trees_mlcourseai.ipynb`, `ml_classification_svm_Vanderplas.ipynb`
3. **Regression** — `ml_assignment04_linreg_optimization_mlcourseai.ipynb`, `ml_assignment06_regression_wine_mlcourseai.ipynb`
4. **Unsupervised** — `ml_assignment07_unsupervised_learning_mlcourseai.ipynb`
5. **Time Series** — `ml_assignment09_time_series_mlcourseai.ipynb`
6. **End-to-end projects** — Airbnb, Ad Demand, Credit Scoring notebooks

## Data
Raw datasets live in the `data/` subdirectory.
