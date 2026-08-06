#!/bin/bash

# Create 19 individual session content files

sessions=(
"1|AI_Overview|AI Overview|ai_intro_tech"
"2|Python_Overview|Python|python_overview"
"3|EDA_DataPrep|EDA|ml_eda_intro,data_preparation_short,ml_eda_endtoend_churn"
"4|Pandas|Pandas|python_intro_pandas"
"5|ML_Intro|ML Intro|ml_intro_short"
"6|ML_Concepts|ML Concepts|ml_concepts_short"
"7|Sklearn_Workflow|Sklearn Workflow|ml_intro_sklearn,ml_datapreparation_sklearn,ml_evaluation_sklearn"
"8|Linear_Regression|LinRegr|ml_linearregression"
"9|Logistic_Regression|LogiRegr|ml_logisticregression"
"10|Decision_Trees|DecTree|ml_decisiontree_short"
"11|Ensemble_RF|Ensemble|ml_ensemble,ml_randomforest"
"12|SVM|SVM|ml_svm"
"13|Naive_Bayes|NaiveBayes|ml_naivebayes_short"
"14|KNN|KNN|ml_knn,ml_knn_sklearn"
"15|KMeans|KMeans|ml_kmeans"
"16|PCA|PCA|ml_pca"
"17|Titanic_Capstone|Titanic|ml_titanic_sklearn"
"18|MLOps_Deployment|MLOps|ml_production,ml_predictive_analytics"
"19|ME_Apps|ME Apps|ml_mech_short,ml_course_demo_regression_housing,ml_course_demo_svm_digits,ml_course_demo_clustering_customers,ml_course_assign_pca_digits,ml_mech_assignments"
)

for session in "${sessions[@]}"; do
    IFS='|' read -r num short section inputs <<< "$session"
    
    filename="course_mlcoep_session_${num}_content.tex"
    
    cat > "$filename" << EOFCONTENT
% Session $num: $section
\section[$section]{Session $num: $section}
EOFCONTENT
    
    IFS=',' read -ra input_array <<< "$inputs"
    for input in "${input_array[@]}"; do
        echo "\input{${input}}" >> "$filename"
    done
    
    echo "Created $filename"
done

echo "Done: 19 session content files created"
