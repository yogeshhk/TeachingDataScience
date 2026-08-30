# Feature Selection Dataset (Session 8)

**Purpose**: Offline copy of the dataset used in the Session 8 hands-on pipeline exercise
(`LaTeX/ml_pipelineexercise.tex`), so the session runs without internet access.

## File

| File | Rows | Columns | Format |
|---|---|---|---|
| `Faults.NNA` | 1941 | 34 (27 features + 7 fault-type indicators) | Whitespace-separated, no header |

Source: UCI Machine Learning Repository, "Steel Plates Faults" dataset
(`https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA`).

## Columns

No header row — supply column names explicitly:

```python
feature_names = [
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
    'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas',
]
fault_names = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
```

The 7 fault-type columns are mutually exclusive binary indicators (each row sums to exactly 1
across them) — i.e. this is really a 7-class problem stored as one-hot columns. The deck uses
`K_Scatch` as a binary target (391 positive / 1550 negative, ~20% positive rate) rather than the
full 7-class problem, to keep the same binary-classification narrative used earlier in the session.

## Usage

```python
import pandas as pd

names = feature_names + fault_names
df = pd.read_csv('Faults.NNA', sep=r'\s+', header=None, names=names)
df['target'] = df['K_Scatch']
```

## Verification

Verified against `pandas`/`scikit-learn` in the `genai` conda env: 1941 rows, 0 missing values,
7 fault columns confirmed mutually exclusive. Full pipeline re-run end to end (load, split, scale,
baseline, filter selection, wrapper greedy-forward selection, final evaluation) reproduces every
number on the slides exactly, including the 5-fold CV accuracies (all 27 features: 0.975, filter
top-10: 0.949, wrapper greedy-6: 0.963) and the final confusion matrix `[[384, 4], [16, 82]]`.

Note: `Bumps` was tried first as the binary target and rejected — greedy forward selection
degenerated to always predicting the majority class (every step plateaued at exactly the 79.3%
majority-class baseline, confusion matrix `[[385, 0], [101, 0]]`). `K_Scatch` gives a real,
cleanly-separable classification problem instead, which is why it's the target used here.
