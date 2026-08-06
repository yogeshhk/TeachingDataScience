# Extract session structure from course_mlcoep_content.tex and create individual session files

$sessions = @(
    @{num=1; short="AI_Overview"; section="AI Overview"; inputs="ai_intro_tech"}
    @{num=2; short="Python_Overview"; section="Python"; inputs="python_overview"}
    @{num=3; short="EDA_DataPrep"; section="EDA"; inputs="ml_eda_intro,data_preparation_short,ml_eda_endtoend_churn"}
    @{num=4; short="Pandas"; section="Pandas"; inputs="python_intro_pandas"}
    @{num=5; short="ML_Intro"; section="ML Intro"; inputs="ml_intro_short"}
    @{num=6; short="ML_Concepts"; section="ML Concepts"; inputs="ml_concepts_short"}
    @{num=7; short="Sklearn_Workflow"; section="Sklearn Workflow"; inputs="ml_intro_sklearn,ml_datapreparation_sklearn,ml_evaluation_sklearn"}
    @{num=8; short="Linear_Regression"; section="LinRegr"; inputs="ml_linearregression"}
    @{num=9; short="Logistic_Regression"; section="LogiRegr"; inputs="ml_logisticregression"}
    @{num=10; short="Decision_Trees"; section="DecTree"; inputs="ml_decisiontree_short"}
    @{num=11; short="Ensemble_RF"; section="Ensemble"; inputs="ml_ensemble,ml_randomforest"}
    @{num=12; short="SVM"; section="SVM"; inputs="ml_svm"}
    @{num=13; short="Naive_Bayes"; section="NaiveBayes"; inputs="ml_naivebayes_short"}
    @{num=14; short="KNN"; section="KNN"; inputs="ml_knn,ml_knn_sklearn"}
    @{num=15; short="KMeans"; section="KMeans"; inputs="ml_kmeans"}
    @{num=16; short="PCA"; section="PCA"; inputs="ml_pca"}
    @{num=17; short="Titanic_Capstone"; section="Titanic"; inputs="ml_titanic_sklearn"}
    @{num=18; short="MLOps_Deployment"; section="MLOps"; inputs="ml_production,ml_predictive_analytics"}
    @{num=19; short="ME_Apps"; section="ME Apps"; inputs="ml_mech_short,ml_course_demo_regression_housing,ml_course_demo_svm_digits,ml_course_demo_clustering_customers,ml_course_assign_pca_digits,ml_mech_assignments"}
)

foreach ($s in $sessions) {
    $contentFile = "course_mlcoep_session_$($s.num)_content.tex"
    $inputLines = ($s.inputs -split ',') | ForEach-Object { "\input{$_.trim()}" }
    $content = "% Session $($s.num): $($s.section)`n\section[$($s.section)]{Session $($s.num): $($s.section)}`n" + ($inputLines -join "`n")
    
    $content | Out-File -FilePath $contentFile -Encoding UTF8
    Write-Host "Created $contentFile"
}

Write-Host "Done: 19 session content files created"
