# MLCoEP TODO (as of 2026-09-19, after the review pass, fixes and decisions)

Items marked (decision) need a call from the instructor before work starts. Items marked (compile) need
approval before any LaTeX compile.

## A. Offline datasets and runnable code (Code/mlcoep): done

Sessions 11, 12, 13 and 14 have no runnable code in their slides; decision: leave them without demos.

Done and smoke-tested in the conda env `genai` (pandas 2.2.3, scikit-learn 1.7.2):

- Datasets with a README each: `shared`, `session09_regression`, `session15_knn`, `session16_kmeans`, `session18_titanic`.
- Scripts for Sessions 9, 10, 15, 16, 17, 18, 19 and 20 (Session 20 reproduces every number on its slides; its README lists the
  external dataset links for the mechanical and agriculture assignments).
- `README_AIML4ME_2026.md` covers Sessions 4 to 20; `environment.yml` has the Session 19 packages, `scikit-learn>=1.2` and
  `python=3.11`; `sessions/.gitignore` ignores `*.png` and `__pycache__/`.

Left over:

1. `environment.yml` has never been built as a whole. A fresh-env build was tried and stopped on purpose (decision: run the scripts in
   `genai`); the `python=3.11` pin is untested.
2. `fetch_california_housing` downloads on first use (internet needed once).
3. Session 20 assignments need packages outside `environment.yml`: `statsmodels`, `deap`, `keras`/`tensorflow` (noted in its README).

## B. Decks (LaTeX/): all delivered to Publications/Presentations on 2026-09-19

Sessions 1, 3, 5, 6, 7, 11, 12, 15 to 20 were recompiled, checked and redelivered. Pages now: S16 63, S17 69, S18 66, S19 62, S20 70.
The full `Main_Course_MachineLearning` driver was left alone (mega-deck).

Done in this round: the flow review of Sessions 16 to 20 and its fixes; the retired "Nutshell" PCA frame (the "Ce, Cf" symbols); the
TikZ redraw of "The Transformed Data"; regenerated Age plots for Session 18; a Netflix opening, a lab, open-source and commercial
platform slides, a quiz, and a career section for Session 19; a quiz per demo in Session 20; the `cars.csv` source (SAS
`SASHELP.CARS`) credited on slides and in the README; the unused images `titk14`, `titk15`, `pca16` deleted.

Remaining, none blocking:

1. Before teaching Session 19, re-check tool and certificate names: DP-100 was retired in June 2026 (successor AI-300), AWS moves its
   ML Engineer exam from MLA-C01 to MLA-C02 in September 2026. The LLMOps block stays commented out (students have not met LLMs).
2. Session 20: open the Kaggle "search" links in a browser (Kaggle blocks scripts). The concrete-strength assignment uses the UCI
   Concrete Compressive Strength data, which matches its title "Strength Prediction".
3. Session 20: the ME assignments still use methods the course does not teach (CNN, LSTM, ARIMA, Gaussian processes, genetic
   algorithms); the divider slide calls them stretch goals. Four demo files repeat the titles "Problem Info" and "Import Libraries",
   which collide in the Beamer outline (harmless).
4. Session 17: the credit "Adapted from: Victor Lavrenko" went away with the retired Nutshell frame; the seven commented-out
   "PCA Pipeline, Step N" frames still carry it if they are ever restored.
5. Decided to keep: the Ng / Wittenauer credit on the Session 16 from-scratch frames.
