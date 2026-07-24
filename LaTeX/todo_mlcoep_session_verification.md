# ML CoEP: Session-by-Session Compile Verification

Working checklist for verifying the rebalanced 19-session `course_mlcoep_content.tex`
compiles cleanly one session at a time, rather than all at once (full-deck compiles are
too slow/noisy to debug from). Delete this file once all 19 rows are checked off, per
the repo's usual `todo_*.md` convention (see other restructuring notes in `CLAUDE.md`).

## Procedure per session

1. In `course_mlcoep_content.tex`, comment out every `\section{}`/`\input{}` block
   **except** the session under test.
2. Run `make.bat` (loops `texify -cp` over `Main_Course_ML_CoEP_*.tex`).
3. Check the `.log` for errors/undefined references/missing images.
4. Compile **both** drivers and rename **both** outputs -- two PDFs per session, kept
   separate (not combined across sessions):
   `Course_MLCoEP_<N>_<ShortName>_Presentation.pdf` and
   `Course_MLCoEP_<N>_<ShortName>_CheatSheet.pdf`.
5. Uncomment the session back, move to the next row.
6. Once all 19 are checked, do one final full-deck compile to confirm nothing broke
   when everything is active together.

## Sessions

- [x] 1 -- AI Overview -- `ai_intro_tech`
- [x] 2 -- Python Overview -- `python_overview`
- [x] 3 -- Understanding Your Data: EDA & Data Prep -- `ml_eda_intro`, `data_preparation_short`, `ml_eda_endtoend_churn`
- [x] 4 -- Doing It: Pandas -- `python_intro_pandas`
- [ ] 5 -- Introduction to Machine Learning -- `ml_intro_short`
- [ ] 6 -- Core ML Concepts -- `ml_concepts_short`
- [ ] 7 -- ML Workflow, Data Prep & Model Evaluation -- `ml_intro_sklearn`, `ml_datapreparation_sklearn`, `ml_evaluation_sklearn`
- [ ] 8 -- Linear Regression -- `ml_linearregression`
- [ ] 9 -- Logistic Regression -- `ml_logisticregression`
- [ ] 10 -- Decision Trees -- `ml_decisiontree_short`
- [ ] 11 -- Ensemble Methods & Random Forest -- `ml_ensemble`, `ml_randomforest`
- [ ] 12 -- Support Vector Machines -- `ml_svm`
- [ ] 13 -- Naive Bayes -- `ml_naivebayes_short`
- [ ] 14 -- K-Nearest Neighbors -- `ml_knn`, `ml_knn_sklearn`
- [ ] 15 -- K-Means Clustering -- `ml_kmeans`
- [ ] 16 -- PCA -- `ml_pca`
- [ ] 17 -- Titanic Capstone -- `ml_titanic_sklearn`
- [ ] 18 -- MLOps & Deployment -- `ml_production`, `ml_predictive_analytics`
- [ ] 19 -- AI/ML Applications & Project Ideas for ME -- `ml_mech_short`, `ml_course_demo_regression_housing`, `ml_course_demo_svm_digits`, `ml_course_demo_clustering_customers`, `ml_course_assign_pca_digits`, `ml_mech_assignments`
- [ ] Final -- full deck, all 19 sessions active together

## Findings so far

- **Pre-existing, repo-wide footer overflow** (found verifying Sessions 3-4): every single
  frame throws an identical `Overfull \hbox (152.98076pt too wide)` warning, regardless of
  content. This is almost certainly the footline template in `template_presentation.tex`
  (`\inserttitle` -- "AI-ML for Mechanical Engineers" -- rendered into a `wd=.5\paperwidth`
  box that isn't wide enough for it) -- constant across frames because it's the same title
  on every slide, not a per-frame content bug. Not caused by, or specific to, the CoEP
  rebalancing work. Not fixed yet -- `template_presentation.tex` is shared across many
  other decks in the repo, so this needs explicit sign-off before touching it. Expect this
  same warning on every remaining session in this checklist; don't mistake it for a new bug.
  **Visually confirmed harmless** (checked pages 1-3 of the Session 3-4 PDF): the footer
  renders cleanly with no clipping or overlap -- LaTeX being conservative about the box
  width, not an actual defect. Safe to ignore this warning throughout the rest of this
  checklist unless a visual check on some other page shows real clipping.

- **`lstlisting` placement violations found and fixed** (Jul 2026): a repo-wide scan for
  content appearing after `\end{lstlisting}` (violates the style rule -- see `upgrade-deck.md`
  Task 1b) found 12 total: 4 in `ml_eda_endtoend_churn.tex` (Session 3), 1 in
  `ml_datapreparation_sklearn.tex` + 7 in `ml_evaluation_sklearn.tex` (both Session 7), and
  1 each in `ml_course_demo_regression_housing.tex`, `ml_course_demo_clustering_customers.tex`,
  `ml_course_assign_pca_digits.tex` (all Session 19). **All 12 now fixed** and re-scan
  confirms zero remaining violations across all 19 sessions' files. Bonus find while fixing
  `ml_evaluation_sklearn.tex`: the `$R^2$ Metric` frame's trailing text was a copy-paste of
  the MSE frame's text and was factually wrong for R\texttt{\^{}}2 (which isn't negated by
  `cross_val_score`, unlike MAE/MSE) -- corrected, not just reordered. Sessions 7 and 19 still
  need their normal compile-and-verify pass (content changed slightly for the R\texttt{\^{}}2
  fix); re-run the placement check on any newly-edited file going forward:
  `python <scratchpad>/check_lstlisting_placement.py <files>` (script written during this
  audit; recreate if no longer in scratchpad).

## Known risk points to watch for while verifying

- Session 3: new content (`ml_eda_intro`, `ml_eda_endtoend_churn`) references images `churn20`/`churn21` -- confirm they render (reused from the existing churn demo's image set).
- Sessions 6, 10, 13 and Session 3's `data_preparation_short`: newly created `_short.tex` comment-siblings -- confirm no frame was cut mid-block (mismatched `\begin{frame}`/`\end{frame}`) by the trimming script.
- Session 19: pulls in 4 files also referenced elsewhere in the repo (shared demo/assign files) -- confirm no name clash or duplicate-label warnings when combined with `ml_mech_short`/`ml_mech_assignments`.
- Appendix "Datasets Used" section references datasets across sessions -- sanity check it still matches after any further session content changes.
