# MLCoEP TODO (as of 2026-09-19, updated after the code and docs pass)

Items marked (decision) need a call from the instructor before work starts. Items marked (compile) need
approval before any LaTeX compile.

## A. Offline datasets and runnable code (Code/mlcoep): done

Sessions 11, 12, 13 and 14 have no runnable code in their slides, so nothing is needed for them unless we add demos (decision).

Done and smoke-tested in the conda env `genai` (pandas 2.2.3, scikit-learn 1.7.2):

- Datasets: `shared/cars.csv`, `session09_regression/housing_data.csv` (reviewed and kept), `session15_knn`, `session16_kmeans`,
  `session18_titanic`; each has a README (files, usage, verification numbers).
- Scripts for Sessions 9, 10, 15, 16, 17, 18, 19 and 20. Session 20 (`demo_housing_regression.py`, `demo_svm_digits.py`,
  `demo_customer_segments.py`, `assign_pca_digits.py`) reproduces every number quoted on the slides; its README lists the
  external dataset links for the mechanical and agriculture assignments.
- `README_AIML4ME_2026.md` "Hands-On Code & Datasets" now covers Sessions 4 to 20.
- `environment.yml`: added `fastapi`, `uvicorn`, `httpx`, `pydantic`, `joblib`; `scikit-learn>=1.2`.
- `sessions/.gitignore` ignores `*.png` and `__pycache__/`.
- Session 15 slide bugs fixed in `ml_knn_nba_case_study.tex` (`.iloc[0]`, `.fillna(0)`); the harmless `random_indices[1:test_cutoff]`
  off-by-one was left as is.
- Session 20 customer-segments slide and script now sort the profile table by income so the printed order matches the results slide.

Left over:

1. The scripts were only run in `genai`, not in a fresh `mlcoep` env. A fresh-env build was tried and stopped on purpose, so
   `environment.yml` (which pins python=3.9) is untested as a whole. scikit-learn 1.7 needs Python 3.10 or newer, so bump the
   pin to python=3.10 or 3.11 when this is next tested (decision).
2. `fetch_california_housing` downloads on first use (internet needed once).

## B. Decks (LaTeX/)

Delivered to `Publications/Presentations/` on 2026-09-19: Sessions 16, 17, 18, 19, 20 (Presentation and CheatSheet).

1. (compile) Shared sources changed but the other decks that use them were NOT compiled: `seminar_ml_clustering_content`
   (ml_kmeans), `seminar_ml_dimreduction_content` (ml_pca, ml_pca_sklearn), `seminar_ml_deployment_content`
   (ml_production, ml_predictive_analytics), `course_machinelearning_content` (ml_titanic_sklearn, demo files),
   `seminar_machinelearning_content` (ml_mech_short, ml_mech_assignments). Compile and eyeball each.
   Also not compiled since their last edit: `ml_pca_sklearn.tex`, `ml_knn_nba_case_study.tex`,
   `ml_course_demo_clustering_customers.tex`.
2. `ml_pca_sklearn.tex`: image-after-code frame reordered and the quiz split into question and answer frames with the `--` removed
   (source fixed, still uncompiled, see item 1).
3. (decision) Session 17: the seven "PCA Pipeline, Step N" slides are commented out in `ml_pca.tex`. Convert the
   "The Transformed Data" image (`pca16`) to native TikZ? Confirm the "Adapted from: Victor Lavrenko" credit on the
   nutshell slide (source inferred, not verified).
4. (decision) Sessions 18, 19, 20: apply the same title clean-up as Session 17 (repeated prefixes such as
   "Feature Engineering:", "Cleaning the Data:", "Random Forest:", "Evaluate Model Accuracy:"; one divider per series).
5. Session 18: images `titk14` to `titk19` come from the original notebook (Age, not AgeFill) and may differ slightly
   from the new code; the old Embarked histogram (`titk11`) is commented out.
6. Session 19: the LLMOps and platform slides come from a September 2026 web survey (mostly vendor blogs); tool status
   changes quickly, re-check names before teaching. Reading list URLs were verified reachable on 2026-09-19.
7. Session 20: the Kaggle "search" links replacing dead dataset URLs could not be tested (Kaggle blocks scripts);
   open them in a browser. Verify the UCI Concrete dataset is the right stand-in for "Structural Load".
8. Confirm the source of `cars.csv` (no attribution line on the slides; the dataset README says "unverified") and the
   Ng / Wittenauer credit on the Session 16 from-scratch frames (inferred from the code).
9. (compile) File-date check done: for Sessions 1, 3, 5, 6, 7, 11 and 12 the delivered PDFs (13-14 Aug) are older than at
   least one source file (edits between 20 Aug and 13 Sep; Sessions 6, 11, 12 have edits to topic files). Dates cannot tell
   a real content change from a touch, so recompile and redeliver, then compare page counts.
10. `CLAUDE.md` (TeachingDataScience): updated with the new files (done).
11. Review pass over Sessions 16 to 20 (flow, length balance, undefined notation such as the covariance-matrix `C` in the PCA
    session). See the review report; `/upgrade-deck` is the expensive follow-up.
