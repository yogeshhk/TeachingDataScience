# TODO: Generate all 19 ML CoEP session PDFs (2 files each)

Task: run the `prep-mlcoep-session` pipeline for Sessions 1-19 of
`course_mlcoep_content.tex`, producing `Course_MLCoEP_<N>_<ShortName>_{Presentation,CheatSheet}.pdf`
for each. Per the user's instruction (2026-08-06): **skip Step 5 (`/upgrade-deck`)** for this pass,
just compile and rename as-is content. Paused mid-run to switch to Haiku model and prioritize other
work; resume later.

## Status

| Session | Dash cleanup | Compiled | Renamed PDFs |
|---|---|---|---|
| 1: AI Overview | done (`ai_intro_tech.tex`) | done | `Course_MLCoEP_1_AI_Overview_{Presentation,CheatSheet}.pdf` |
| 2: Python Overview | done (`python_overview.tex`) | done | `Course_MLCoEP_2_Python_Overview_{Presentation,CheatSheet}.pdf` |
| 3: EDA DataPrep | **not started** | no | no |
| 4: Pandas | **not started** | no | no |
| 5: ML Intro | **not started** | no | no |
| 6: ML Concepts | **not started** | no | no |
| 7: Sklearn Workflow | **not started** | no | no |
| 8: Linear Regression | **not started** | no | no |
| 9: Logistic Regression | **not started** | no | no |
| 10: Decision Trees | **not started** | no | no |
| 11: Ensemble RF | **not started** | no | no |
| 12: SVM | **not started** | no | no |
| 13: Naive Bayes | **not started** | no | no |
| 14: KNN | **not started** | no | no |
| 15: KMeans | **not started** | no | no |
| 16: PCA | **not started** | no | no |
| 17: Titanic Capstone | **not started** | no | no |
| 18: MLOps Deployment | **not started** | no | no |
| 19: ME Apps | **not started** | no | no |

`course_mlcoep_content.tex` is currently in its normal all-19-active resting state (appendix live) --
confirmed restored after stopping. Safe to resume from Session 3 without any file-state cleanup first.

## Dash cleanup already surveyed (saves a re-grep when resuming)

A one-time grep across every topic file used by all 19 sessions found these files still have
literal `--` prose dashes needing cleanup (colon/comma per the standing convention), beyond the 2
already fixed above. Numeric ranges and table N/A placeholders were excluded from this list already:

- `data_preparation_short.tex` (Session 3) -- 3 instances (~lines 27, 30, 497)
- `ml_concepts_short.tex` (Session 6) -- 5 instances (~lines 238, 1044, 1318 [commented, skip], 1392 [commented, skip], 1926)
- `ml_intro_short.tex` (Session 5) -- 6 instances (~lines 133, 314, 375, 592, 1729, 1793)
- `ml_datapreparation_sklearn.tex` (Session 7) -- 2 instances (~lines 149, 204)
- `ml_logisticregression.tex` (Session 9) -- 1 instance (~line 521, but check: may be in a commented frame)
- `ml_evaluation_sklearn.tex` (Session 7) -- 1 instance (~line 282)
- `ml_ensemble.tex` (Session 11) -- 2 instances (~lines 98, 158)
- `ml_decisiontree_short.tex` (Session 10) -- 2 instances (~lines 1237 [has 2 dashes in one line], 1900)
- `ml_randomforest.tex` (Session 11) -- 1 instance (~line 110)
- `ml_naivebayes_short.tex` (Session 13) -- 2 instances (~lines 539, 1071)
- `ml_kmeans.tex` (Session 15) -- 3 instances (~lines 395, 479, 777)
- `ml_svm.tex` (Session 12) -- 2 instances (~lines 1186, 1189, same sentence spanning 2 lines)
- `ml_pca.tex` (Session 16) -- 1 instance (~line 544)
- `ml_titanic_sklearn.tex` (Session 17) -- 1 instance (~line 514, uses a plain `--` with no spaces, "feature--unfortunately")
- `ml_knn_sklearn.tex` (Session 14) -- 1 instance (~line 402)
- `ml_predictive_analytics.tex` (Session 18) -- 2 instances (~lines 624-625, spans 2 lines)

Re-verify line numbers before editing (they may have shifted if anything else touched these files
since 2026-08-06). Session 4's `python_intro_pandas.tex`, Session 8's `ml_linearregression.tex`,
Session 19's topic files, and the rest of Session 18/13's files had no prose-dash hits in this survey.

## REPL disambiguation (Step 3) already checked: nothing needed

Grepped every topic file for `Out[`/`In[` cell-number-label patterns (the ambiguity trigger for
Step 3). Zero matches anywhere except the already-fixed `python_intro_pandas.tex` (Session 4, fixed
in an earlier pass per `CLAUDE.md`'s ML CoEP note). Safe to skip Step 3 entirely when resuming
Sessions 3-19 unless a closer read of a specific file's `lstlisting` blocks finds something the
`Out[`/`In[` grep pattern didn't catch.

## How to resume

Either invoke `/prep-mlcoep-session <N>` normally for each remaining session (skip its Step 5
prompt, same as this pass), or recreate the isolate/restore helper: `course_mlcoep_content.tex`'s 19
`\section{Session N: ...}` blocks + one `\appendix` block can be toggled by commenting out every
block except the target with `%`-prefixing per line, then restoring by uncommenting all. Compile via
`LaTeX/make.bat` (loops `texify -cp` over `Main_Course_ML_CoEP_*.tex`), then rename
`Main_Course_ML_CoEP_Presentation.pdf` / `_CheatSheet.pdf` to
`Course_MLCoEP_<N>_<ShortName>_{Presentation,CheatSheet}.pdf` using each session's title text after
"Session N:" with spaces replaced by underscores (e.g. Session 3 = `EDA_DataPrep`).

Delete this file once all 19 sessions are done, per the repo's usual `todo_*.md` convention.
