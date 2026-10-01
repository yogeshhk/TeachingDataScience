# In-Class Test: Crop Recommendation 

## Task

Given soil and climate measurements for a field, predict which crop suits it. Read the data, understand it, explore
it, prepare it, train a classifier, and evaluate it. Work in a Jupyter notebook.

## Data card

**File**: `Crop_recommendation.csv` (comma-separated, 2200 rows, 8 columns, no missing values).
**Source**: Kaggle, "Crop Recommendation" data set.

Each row is one field. The first seven columns are the inputs (features) and the last column is the answer (label).

| Column | Type | Meaning |
|---|---|---|
| `N` | integer | Nitrogen content in the soil |
| `P` | integer | Phosphorus content in the soil |
| `K` | integer | Potassium content in the soil |
| `temperature` | float | Temperature in degrees Celsius |
| `humidity` | float | Relative humidity in percent |
| `ph` | float | Soil pH (acidity), roughly 3.5 to 10 |
| `rainfall` | float | Rainfall in mm |
| `label` | text | **The crop to predict**, for example `rice` or `maize` |

There are 22 crops (apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil,
maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon) with 100 rows
each. Predicting `label` from the seven features is a **multiclass classification** problem.

## Steps

Every step needs working code, its output visible in the notebook, and one or two lines of your own comment
(markdown cell) saying what you see or why you did it.

| # | Step | What is expected |
|---|---|---|
| 1 | Read and understand the data | Load the CSV; show shape, first rows, `info()`, `describe()`; say what the features and the label are |
| 2 | Pre-processing | Check missing values, duplicates, data types and class balance; state what you found |
| 3 | Distribution plots | Histogram or box plot for each feature; comment on scale and skew |
| 4 | Correlation plot | Correlation heatmap of the features; point out strongly correlated pairs |
| 5 | Remove correlated features | Drop one feature from each strongly correlated pair (say the threshold you used and why) |
| 6 | Standardization or normalization | Scale the features (for example `StandardScaler` or `MinMaxScaler`); fit the scaler on the training data only |
| 7 | Split and train | Train and test split (stratified is better), then train a classifier: Logistic Regression, Naive Bayes, or any other |
| 8 | Metrics | Accuracy, precision, recall and F1 on the test set (macro average for the multiclass case); a confusion matrix or classification report is a good addition |
| 9 | K-fold cross-validation | Stratified K-fold (for example 5-fold) with the same metrics; report mean and spread |
| 10 | Comparison and conclusion | Compare at least two models, say which is better and why |

Attempt every step. If time is short, finish steps 1 to 8 first and do the K-fold and comparison steps last.

## What to submit

1. Save your notebook as `<your Name-MIS number>_AIML_BTECH_Final.ipynb`, for example `YogeshKulkarni-112345678_AIML_BTECH_Final.ipynb`.
2. Run **Kernel, Restart and Run All** once before saving, so every output is visible and the notebook runs without errors.
3. Email the `.ipynb` file to the instructor and to the second examiner before the time ends:
   - Instructors: `yhk.mech and cc ssp.mech`
   - Subject line: `AIML_BTECH_Final_ClassTest-Oct2026`

A notebook that errors out part-way is assessed only up to the cell that fails.

## Setup: install these before the test

Use Python 3.10 or newer. The minimum packages (versions shown were checked to work; newer ones are fine):

| Package | Version |
|---|---|
| `pandas` | 2.2 |
| `numpy` | 1.26 or newer |
| `matplotlib` | 3.8 or newer |
| `seaborn` | 0.12 or newer |
| `scikit-learn` | 1.3 or newer |
| `jupyter` (or `notebook` / `jupyterlab`) | any recent |

Install in your conda environment, then check the imports work:

```
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('ok')"
```

Put `Crop_recommendation.csv` in the same folder as your notebook, so `pd.read_csv("Crop_recommendation.csv")` works.

Things that can look odd but are not your mistake:

- Start Jupyter from an **activated** conda environment (Anaconda Prompt, then `conda activate <env>`, then `jupyter notebook`).
  A kernel started outside an activated environment can fail with `DLL load failed while importing _multiarray_umath`.
- In some environments, the first `import pandas` prints a `DLL load failed ... _multiarray_umath` message. It comes
  from the optional `numexpr` package and pandas continues without it. If it bothers you, run
  `conda install -U numexpr` (or `pip install -U numexpr`).
