# ML Course Seminar Upgrade — TODO

Scope: 10 new seminar content files created for the ML course.
Strategy: compile each seminar standalone to verify it builds, then run /upgrade-deck on it.
Assumption: if all seminars are upgraded, the workshops and the full course will compile correctly.

Driver files (Presentation + CheatSheet) for all 10 seminars have already been created.

---

## Seminar 1 — ML Introduction
Content file : seminar_ml_intro_content.tex
Driver       : Main_Seminar_ML_Intro_Presentation.tex
Covers       : ml_intro, ml_concepts, ml_intro_sklearn

- [ ] Compile   : `texify -cp Main_Seminar_ML_Intro_Presentation.tex`
- [x] Upgrade   : `/upgrade-deck Main_Seminar_ML_Intro_Presentation.tex`

---

## Seminar 2 — Data Preparation
Content file : seminar_ml_dataprep_content.tex
Driver       : Main_Seminar_ML_DataPrep_Presentation.tex
Covers       : python_intro_pandas, ml_datapreparation_sklearn, ml_evaluation_sklearn

- [ ] Compile   : `texify -cp Main_Seminar_ML_DataPrep_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_DataPrep_Presentation.tex`

---

## Seminar 3 — Regression
Content file : seminar_ml_regression_content.tex
Driver       : Main_Seminar_ML_Regression_Presentation.tex
Covers       : ml_linearregression, ml_linearregression_sklearn, ml_regularization,
               ml_logisticregression, ml_logisticregression_sklearn

- [ ] Compile   : `texify -cp Main_Seminar_ML_Regression_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_Regression_Presentation.tex`

---

## Seminar 4 — Decision Trees
Content file : seminar_ml_decisiontree_content.tex
Driver       : Main_Seminar_ML_DecisionTree_Presentation.tex
Covers       : ml_decisiontree, ml_decisiontree_sklearn

- [ ] Compile   : `texify -cp Main_Seminar_ML_DecisionTree_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_DecisionTree_Presentation.tex`

---

## Seminar 5 — Ensemble Methods and Random Forest
Content file : seminar_ml_ensemble_content.tex
Driver       : Main_Seminar_ML_Ensemble_Presentation.tex
Covers       : ml_ensemble, ml_ensemble_sklearn, ml_randomforest, ml_randomforest_sklearn

- [ ] Compile   : `texify -cp Main_Seminar_ML_Ensemble_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_Ensemble_Presentation.tex`

---

## Seminar 6 — K-Nearest Neighbor
Content file : seminar_ml_knn_content.tex
Driver       : Main_Seminar_ML_KNN_Presentation.tex
Covers       : ml_knn, ml_knn_sklearn

- [ ] Compile   : `texify -cp Main_Seminar_ML_KNN_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_KNN_Presentation.tex`

---

## Seminar 7 — Support Vector Machines and Naive Bayes
Content file : seminar_ml_svm_nb_content.tex
Driver       : Main_Seminar_ML_SVM_NB_Presentation.tex
Covers       : ml_svm, ml_svm_sklearn, ml_naivebayes, ml_naivebayes_sklearn

- [ ] Compile   : `texify -cp Main_Seminar_ML_SVM_NB_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_SVM_NB_Presentation.tex`

---

## Seminar 8 — Clustering
Content file : seminar_ml_clustering_content.tex
Driver       : Main_Seminar_ML_Clustering_Presentation.tex
Covers       : ml_kmeans, ml_kmeans_sklearn

- [ ] Compile   : `texify -cp Main_Seminar_ML_Clustering_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_Clustering_Presentation.tex`

---

## Seminar 9 — Dimensionality Reduction
Content file : seminar_ml_dimreduction_content.tex
Driver       : Main_Seminar_ML_DimReduction_Presentation.tex
Covers       : ml_pca, ml_pca_sklearn, ml_svd

- [ ] Compile   : `texify -cp Main_Seminar_ML_DimReduction_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_DimReduction_Presentation.tex`

---

## Seminar 10 — Production and Deployment
Content file : seminar_ml_deployment_content.tex
Driver       : Main_Seminar_ML_Deployment_Presentation.tex
Covers       : ml_production, ml_predictive_analytics, ml_conclusion,
               ml_conclusion_sklearn, ml_refs

- [ ] Compile   : `texify -cp Main_Seminar_ML_Deployment_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_ML_Deployment_Presentation.tex`

---

## After all seminars done

- [ ] Compile full course  : `texify -cp Main_Course_MachineLearning_Presentation.tex`
- [ ] Compile ML workshop  : `texify -cp Main_Workshop_MachineLearning_Presentation.tex`
