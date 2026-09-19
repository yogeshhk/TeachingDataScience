# MLCoEP TODO (as of 2026-09-19, after the review pass and fixes)

Items marked (decision) need a call from the instructor before work starts. Items marked (compile) need
approval before any LaTeX compile.

## A. Offline datasets and runnable code (Code/mlcoep): done

Sessions 11, 12, 13 and 14 have no runnable code in their slides, so nothing is needed for them unless we add demos (decision).

Done and smoke-tested in the conda env `genai` (pandas 2.2.3, scikit-learn 1.7.2):

- Datasets with a README each: `shared`, `session09_regression`, `session15_knn`, `session16_kmeans`, `session18_titanic`.
- Scripts for Sessions 9, 10, 15, 16, 17, 18, 19 and 20 (Session 20 reproduces every number on its slides; its README lists the
  external dataset links for the mechanical and agriculture assignments).
- `README_AIML4ME_2026.md` covers Sessions 4 to 20; `environment.yml` has the Session 19 packages, `scikit-learn>=1.2` and now
  `python=3.11`; `sessions/.gitignore` ignores `*.png` and `__pycache__/`.

Left over:

1. `environment.yml` has never been built as a whole. A fresh-env build was tried and stopped on purpose; the scripts were only run in
   `genai`. Build `mlcoep` once and re-run the scripts when convenient.
2. `fetch_california_housing` downloads on first use (internet needed once).
3. Session 20 assignments need packages outside `environment.yml`: `statsmodels`, `deap`, `keras`/`tensorflow` (noted in its README).

## B. Decks (LaTeX/)

Done on 2026-09-19:

- Compiled and checked the shared-source decks (Clustering, DimReduction, Deployment seminars) and Session 15.
- Recompiled and redelivered Sessions 1, 3, 5, 6, 7, 11, 12 (page counts rose, so the content had changed).
- Review pass over Sessions 16 to 20 (flow, length, undefined notation, correctness) and its fixes, then recompiled, eyeballed the
  changed pages and redelivered Sessions 16 to 20 to `Publications/Presentations/` (pages before to after: S16 69 to 63,
  S17 77 to 69, S18 81 to 66, S19 95 to 37, S20 76 to 64).
- The "Ce, Cf" symbols were the covariance matrix in the "In a Nutshell" PCA diagram; that frame is retired. Cov(x,y), the T/A/C
  letters and the d, k notation are now defined in the PCA deck; "The Transformed Data" is native TikZ.
- Session 18 `titk14` and `titk15` were built from `Age`, not `AgeFill`; regenerated as `images/titk14_agefill.png` and
  `titk15_agefill.png` (the other Age images already matched the slide code).

Not compiled after their last edit: none of the edited decks; the full `Main_Course_MachineLearning` driver was left alone (mega-deck).

Remaining:

1. (decision) S19 is now 37 pages: `ml_predictive_analytics` and the LLMOps block are out. If that is too thin, bring back the
   Netflix "why deployment matters" frames (about 6) from `ml_predictive_analytics.tex` as an opening.
2. (decision) The demos in S20 (housing, SVM, customers) have no quiz, and the ME assignments use methods the course does not
   teach (noted on the divider slide). Add a quiz per demo, or trim the assignment list?
3. The old images `titk14.png`, `titk15.png` and `pca16.png` are no longer referenced; delete them when convenient.
4. Session 19: the platform-landscape frames are now optional (commented out). If you restore them, re-check the tool names first;
   the reading list URLs were verified reachable on 2026-09-19.
5. Session 20: the Kaggle "search" links could not be tested (Kaggle blocks scripts); open them in a browser. Verify the UCI
   Concrete dataset is the right stand-in for "Structural Load" (the slide is now titled "Strength Prediction").
6. Confirm the source of `cars.csv` (no attribution line; the dataset README says "unverified") and the Ng / Wittenauer credit on the
   Session 16 from-scratch frames (inferred from the code).
7. Session 17 credit "Adapted from: Victor Lavrenko" went away with the retired Nutshell frame; the seven commented-out
   "PCA Pipeline, Step N" frames still carry it if they are ever restored.
8. Session 20 titles: four demo files repeat "Problem Info" and "Import Libraries"; harmless, but they collide in the Beamer outline.
