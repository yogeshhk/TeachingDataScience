"""
Session 8 (Feature Selection) hands-on pipeline, extracted end-to-end from
LaTeX/ml_pipelineexercise.tex ("Hands-On: End-to-End Mini Pipeline", Steps 1-11).

Dataset: UCI Steel Plates Faults, local offline copy at
Code/mlcoep/datasets/session08_featureselection/Faults.NNA
Target: K_Scatch (scratch-type surface defect) vs. everything else.

Run: conda activate genai && python pipeline_steel_plates_faults.py
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets",
    "session08_featureselection", "Faults.NNA",
)

# --- Step 1: Load and Look ---
feature_names = [
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
    'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas',
]
fault_names = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
               'Dirtiness', 'Bumps', 'Other_Faults']

df = pd.read_csv(DATA_PATH, sep=r'\s+', header=None,
                  names=feature_names + fault_names)
df['target'] = df['K_Scatch']
print("Step 1: Load and Look")
print(df.shape)
print(df.head(3))
print()

# --- Step 2: Missing Values and Class Balance ---
print("Step 2: Missing Values and Class Balance")
print(df.isna().sum().sum(), "missing values")
print(df['target'].value_counts())
print()

# --- Step 3: Train/Test Split ---
X = df[feature_names].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print("Step 3: Train/Test Split")
print("X_train", X_train.shape, "  X_test", X_test.shape)
print()

# --- Step 4: Scale the Features ---
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Step 5: Baseline, All 27 Features ---
model_all = LogisticRegression(max_iter=5000)
model_all.fit(X_train_s, y_train)
pred_all = model_all.predict(X_test_s)
print("Step 5: Baseline, All 27 Features")
print("Accuracy:", accuracy_score(y_test, pred_all))
print()

# --- Step 6: Rank Features (Filter) ---
skb = SelectKBest(score_func=f_classif, k=10).fit(X_train_s, y_train)
top10 = [feature_names[i] for i in skb.get_support(indices=True)]
print("Step 6: Rank Features (Filter)")
print(top10)
print()

# --- Step 7: Retrain on the Top 10 ---
idx = skb.get_support(indices=True)
model_10 = LogisticRegression(max_iter=5000)
model_10.fit(X_train_s[:, idx], y_train)
pred_10 = model_10.predict(X_test_s[:, idx])
print("Step 7: Retrain on the Top 10")
print("Accuracy:", accuracy_score(y_test, pred_10))
print()

# --- Step 8: Filter Result -- no code, slide-only comparison ---

# --- Step 9: Try a Wrapper Instead ---
Xs = StandardScaler().fit_transform(X)
kfold = KFold(n_splits=5, shuffle=True, random_state=7)
selected, remaining = [], list(range(27))

for step in range(6):
    scores = [(f, cross_val_score(LogisticRegression(max_iter=5000),
               Xs[:, selected + [f]], y, cv=kfold).mean()) for f in remaining]
    best_f, best_s = max(scores, key=lambda t: t[1])
    selected.append(best_f)
    remaining.remove(best_f)

print("Step 9: Try a Wrapper Instead")
print("Selected (in order):", [feature_names[i] for i in selected])
print()

# --- Step 10: Filter vs. Wrapper vs. Everything ---
acc_all27 = cross_val_score(LogisticRegression(max_iter=5000), Xs, y, cv=kfold).mean()
acc_filter10 = cross_val_score(LogisticRegression(max_iter=5000), Xs[:, idx], y, cv=kfold).mean()
acc_wrapper6 = cross_val_score(LogisticRegression(max_iter=5000), Xs[:, selected], y, cv=kfold).mean()
print("Step 10: Filter vs. Wrapper vs. Everything (5-fold CV accuracy)")
print("All 27 features:            ", round(acc_all27, 3))
print("Filter (top 10 by F-score): ", round(acc_filter10, 3))
print("Wrapper (greedy forward 6): ", round(acc_wrapper6, 3))
print()

# --- Step 11: Evaluate the Final Model ---
wrapper_cols = [feature_names[i] for i in selected]
idx6 = [feature_names.index(f) for f in wrapper_cols]

model_6 = LogisticRegression(max_iter=5000)
model_6.fit(X_train_s[:, idx6], y_train)
pred_6 = model_6.predict(X_test_s[:, idx6])
print("Step 11: Evaluate the Final Model")
print("Wrapper columns:", wrapper_cols)
print(confusion_matrix(y_test, pred_6))
print()
print(classification_report(y_test, pred_6,
                             target_names=['no_k_scatch', 'k_scatch']))
