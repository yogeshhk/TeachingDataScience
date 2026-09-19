# MLCoEP TODO (as of 2026-09-19)

Everything below is pending. Items marked (decision) need a call from the instructor before work starts.

## A. Offline datasets and runnable code (Code/mlcoep), half done

Sessions 9 to 20 previously had no code or data here (only Sessions 4, 7, 8). Sessions 11, 12, 13 and 14
have no runnable code in their slides, so nothing is needed for them unless we add demos (decision).

Done and smoke-tested with pandas 2.2.3 / scikit-learn 1.7.2 (conda env `genai`):

- `datasets/shared/cars.csv` (used by Sessions 16, 17, 19), `session15_knn/nba_2013.csv`,
  `session16_kmeans/ex7data2.mat`, `session18_titanic/titanic_{train,test}.csv`
  (all copied from `Code/ml/data`, byte-identical)
- `sessions/session09_linear_regression/` `make_housing_data.py`, `house_price_regression.py`
  (housing_data.csv is synthetic and seeded; the smoke test already generated it, 500 rows)
- `sessions/session10_logistic_regression/sigmoid_plot.py`
- `sessions/session15_knn/` `knn_from_scratch.py`, `nba_similar_players.py`
- `sessions/session16_kmeans/` `kmeans_from_scratch.py`, `kmeans_cars_case_study.py`
- `sessions/session17_pca/` `pca_worked_example.py`, `pca_cars_case_study.py`
- `sessions/session18_titanic/titanic_random_forest.py`
- `sessions/session19_mlops/` `train_model.py`, `app.py`, `test_api.py`, `drift_check.py`, `Dockerfile`,
  `requirements.txt`, `.gitignore`

Still to do:

1. Review the generated `datasets/session09_regression/housing_data.csv` and keep it (regenerate with `make_housing_data.py`;
   the deck's model gives R2 0.951 only after adding the sqft x neighborhood interaction, see `house_price_regression.py`).
2. Session 20 scripts, not written yet, from the deck code in `LaTeX/ml_course_demo_*.tex` and
   `ml_course_assign_pca_digits.tex`: `demo_housing_regression.py`, `demo_svm_digits.py`,
   `demo_customer_segments.py`, `assign_pca_digits.py`. Verified results to reproduce:
   Linear/Ridge RMSE 0.7456 R2 0.5758, Lasso(0.1) RMSE 0.8244 R2 0.4814; SVM digits accuracy 0.9806 (7 errors of 360);
   customer elbow inertia 720.0, 216.9, 87.7, 76.5, 65.2, 54.8, 48.2, 41.4, 37.3.
   `fetch_california_housing` downloads on first use (needs internet once).
   Assignment lists (`ml_mech_assignments`, `ml_agri_assignments`) use external datasets: add a README with the
   links only, no code.
3. Dataset READMEs in the Session 7 format (purpose, files table, usage, verification numbers) for:
   `shared`, `session09_regression`, `session15_knn`, `session16_kmeans`, `session18_titanic`.
4. Update `README_AIML4ME_2026.md` section "Hands-On Code & Datasets": it still says Sessions 4, 7, 8 only.
5. Update `environment.yml`: add `fastapi`, `uvicorn`, `httpx`, `pydantic` (Session 19); raise `scikit-learn`
   to >=1.2 (`n_init='auto'`); the file pins python=3.9, so re-test all scripts in a fresh `mlcoep` env
   (they were only run in `genai`).
6. Add `*.png` for generated figures to a `.gitignore` under `sessions/` (`sigmoid_plot.py` saves `sigmoid.png` on a headless machine).
7. Session 15 slide bugs found while writing `nba_similar_players.py`, fix in `LaTeX/ml_knn_nba_case_study.tex`:
   - block 4: `distance.euclidean(row, lebron_normalized)` fails on current SciPy because `lebron_normalized` is a
     2-D one-row DataFrame; use `.iloc[0]`.
   - block 6: `KNeighborsRegressor.fit(train[x_columns], ...)` fails because 94 cells in those columns are NaN;
     use `.fillna(0)` (the slide only fills the normalized copy).
   - block 5 takes the test rows from `random_indices[1:test_cutoff]`, which skips one row (harmless).

## B. Decks (LaTeX/)

Delivered to `Publications/Presentations/` on 2026-09-19: Sessions 16, 17, 18, 19, 20 (Presentation and CheatSheet).

1. Shared sources changed but the other decks that use them were NOT compiled: `seminar_ml_clustering_content`
   (ml_kmeans), `seminar_ml_dimreduction_content` (ml_pca, ml_pca_sklearn), `seminar_ml_deployment_content`
   (ml_production, ml_predictive_analytics), `course_machinelearning_content` (ml_titanic_sklearn, demo files),
   `seminar_machinelearning_content` (ml_mech_short, ml_mech_assignments). Compile and eyeball each (ask first).
2. `ml_pca_sklearn.tex` was fixed (StandardScaler, per-feature normalization, cost claim, new `images/pca18_recovered.png`)
   but not compiled. It still has one frame with an image after a code block, and a quiz that uses `\pause` and a `--` dash.
3. Session 17 (decision): the seven "PCA Pipeline, Step N" slides are commented out in `ml_pca.tex`. Convert the
   "The Transformed Data" image (`pca16`) to native TikZ? Confirm the "Adapted from: Victor Lavrenko" credit on the
   nutshell slide (source inferred, not verified).
4. Sessions 18, 19, 20 (decision): apply the same title clean-up as Session 17 (repeated prefixes such as
   "Feature Engineering:", "Cleaning the Data:", "Random Forest:", "Evaluate Model Accuracy:"; one divider per series).
5. Session 18: images `titk14` to `titk19` come from the original notebook (Age, not AgeFill) and may differ slightly
   from the new code; the old Embarked histogram (`titk11`) is commented out.
6. Session 19: the LLMOps and platform slides come from a September 2026 web survey (mostly vendor blogs); tool status
   changes quickly, re-check names before teaching. Reading list URLs were verified reachable on 2026-09-19.
7. Session 20: the Kaggle "search" links replacing dead dataset URLs could not be tested (Kaggle blocks scripts);
   open them in a browser. Verify the UCI Concrete dataset is the right stand-in for "Structural Load".
8. Confirm the source of `cars.csv` (no attribution line on the slides) and the Ng / Wittenauer credit on the
   Session 16 from-scratch frames (inferred from the code).
9. Delivered PDFs for Sessions 1, 3, 5, 6, 7, 11, 12 are older than their sources (file-date check, may be a
   false alarm); recompile and redeliver if the content really changed.
10. Update `CLAUDE.md` (TeachingDataScience) with the new files: `ml_mlops_workflow`, `ml_mlops_production`,
    `ml_kmeans_cars_case_study`, `ml_pca_cars_case_study`, and the note that Session 17 has 3 hands-on frames beyond the theory.
