"""
Session 8 (Feature Selection) "Try It Yourself" exercises, from the closing frame of
LaTeX/ml_pipelineexercise.tex. Reuses the same Steel Plates Faults setup as
pipeline_steel_plates_faults.py (Steps 1-4: load, split, scale) and then runs the
four suggested extensions:

  1. k=10 -> k=5 in Step 6: does filter accuracy hold up with even fewer features?
  2. Greedy Backward Elimination instead of Forward on all 27 features: same 6
     features at the end?
  3. Swap LogisticRegression for a different classifier: does the winning feature
     set change?
  4. Try a different target fault type (Bumps): does the same feature set still win?

Run: conda activate genai && python try_it_yourself.py
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets",
    "session08_featureselection", "Faults.NNA",
)

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


def greedy_forward(Xs, y, kfold, n_rounds, clf_factory):
    selected, remaining = [], list(range(len(feature_names)))
    for _ in range(n_rounds):
        scores = [(f, cross_val_score(clf_factory(),
                   Xs[:, selected + [f]], y, cv=kfold).mean()) for f in remaining]
        best_f, _ = max(scores, key=lambda t: t[1])
        selected.append(best_f)
        remaining.remove(best_f)
    return selected


def greedy_backward(Xs, y, kfold, clf_factory):
    selected = list(range(len(feature_names)))
    current_score = cross_val_score(clf_factory(), Xs[:, selected], y, cv=kfold).mean()
    removed_order = []
    while len(selected) > 1:
        best_drop, best_score_after = None, -1
        for f in selected:
            candidate = [c for c in selected if c != f]
            s = cross_val_score(clf_factory(), Xs[:, candidate], y, cv=kfold).mean()
            if s > best_score_after:
                best_score_after, best_drop = s, f
        if best_score_after >= current_score:
            selected.remove(best_drop)
            removed_order.append((best_drop, best_score_after))
            current_score = best_score_after
        else:
            break
    return selected, removed_order


def run_for_target(target_col):
    d = df.copy()
    d['target'] = d[target_col]
    X = d[feature_names].values
    y = d['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    Xs = StandardScaler().fit_transform(X)
    return X, y, X_train_s, X_test_s, y_train, y_test, Xs


if __name__ == "__main__":
    X, y, X_train_s, X_test_s, y_train, y_test, Xs = run_for_target('K_Scatch')
    kfold = KFold(n_splits=5, shuffle=True, random_state=7)

    # --- Exercise 1: k=10 -> k=5 ---
    print("Exercise 1: k=10 vs k=5 (filter, test accuracy)")
    for k in (10, 5):
        skb = SelectKBest(score_func=f_classif, k=k).fit(X_train_s, y_train)
        idx = skb.get_support(indices=True)
        model = LogisticRegression(max_iter=5000).fit(X_train_s[:, idx], y_train)
        acc = accuracy_score(y_test, model.predict(X_test_s[:, idx]))
        cols = [feature_names[i] for i in idx]
        print(f"  k={k}: accuracy={acc:.3f}  features={cols}")
    print()

    # --- Exercise 2: Greedy Backward Elimination on all 27 features ---
    print("Exercise 2: Greedy Backward Elimination (K_Scatch, LogisticRegression)")
    selected_bwd, removed_order = greedy_backward(
        Xs, y, kfold, lambda: LogisticRegression(max_iter=5000))
    print("  Removed in order:", [(feature_names[f], round(s, 3)) for f, s in removed_order])
    print("  Remaining", len(selected_bwd), "features:", [feature_names[i] for i in selected_bwd])
    forward_6 = {'Log_X_Index', 'Steel_Plate_Thickness', 'Pixels_Areas',
                 'Square_Index', 'X_Minimum', 'Orientation_Index'}
    backward_set = {feature_names[i] for i in selected_bwd}
    print("  Same as forward-selected 6?", backward_set == forward_6,
          "  (backward kept", len(backward_set), "features)")
    print()

    # --- Exercise 3: swap classifier ---
    print("Exercise 3: Greedy Forward Selection with RandomForestClassifier")
    selected_rf = greedy_forward(
        Xs, y, kfold, 6, lambda: RandomForestClassifier(n_estimators=100, random_state=7))
    rf_cols = [feature_names[i] for i in selected_rf]
    print("  RandomForest selected (in order):", rf_cols)
    print("  Same as LogisticRegression's 6?", set(rf_cols) == forward_6)
    print()

    # --- Exercise 4: different target (Bumps) ---
    print("Exercise 4: Greedy Forward Selection on target=Bumps")
    Xb, yb, _, _, _, _, Xsb = run_for_target('Bumps')
    print("  Bumps class balance:", pd.Series(yb).value_counts().to_dict())
    selected_bumps = greedy_forward(
        Xsb, yb, kfold, 6, lambda: LogisticRegression(max_iter=5000))
    bumps_cols = [feature_names[i] for i in selected_bumps]
    print("  Bumps selected (in order):", bumps_cols)
    print("  Same as K_Scatch's 6?", set(bumps_cols) == forward_6)
    baseline = 1 - pd.Series(yb).mean()
    final_score = cross_val_score(LogisticRegression(max_iter=5000),
                                   Xsb[:, selected_bumps], yb, cv=kfold).mean()
    print(f"  Majority-class baseline: {baseline:.3f}   Final 6-feature CV accuracy: {final_score:.3f}")
