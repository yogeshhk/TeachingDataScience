@echo off
REM ML CoEP: Compile all 20 sessions (40 driver files: 20 Presentation + 20 CheatSheet)
REM Usage: make_all_sessions.bat
REM Each texify call compiles independently; failures in one do not stop the rest.

echo Compiling all 20 ML CoEP sessions...
echo.

REM Session 1: AI Overview
texify -cp Main_Seminar_MLCoEP_Session_1_AI_Overview_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_1_AI_Overview_CheatSheet.tex

REM Session 2: Python Overview
texify -cp Main_Seminar_MLCoEP_Session_2_Python_Overview_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_2_Python_Overview_CheatSheet.tex

REM Session 3: EDA DataPrep
texify -cp Main_Seminar_MLCoEP_Session_3_EDA_DataPrep_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_3_EDA_DataPrep_CheatSheet.tex

REM Session 4: Pandas
texify -cp Main_Seminar_MLCoEP_Session_4_Pandas_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_4_Pandas_CheatSheet.tex

REM Session 5: ML Intro
texify -cp Main_Seminar_MLCoEP_Session_5_ML_Intro_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_5_ML_Intro_CheatSheet.tex

REM Session 6: ML Concepts
texify -cp Main_Seminar_MLCoEP_Session_6_ML_Concepts_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_6_ML_Concepts_CheatSheet.tex

REM Session 7: Sklearn Workflow
texify -cp Main_Seminar_MLCoEP_Session_7_Sklearn_Workflow_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_7_Sklearn_Workflow_CheatSheet.tex

REM Session 8: Feature Selection
texify -cp Main_Seminar_MLCoEP_Session_8_Feature_Selection_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_8_Feature_Selection_CheatSheet.tex

REM Session 9: Linear Regression
texify -cp Main_Seminar_MLCoEP_Session_9_Linear_Regression_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_9_Linear_Regression_CheatSheet.tex

REM Session 10: Logistic Regression
texify -cp Main_Seminar_MLCoEP_Session_10_Logistic_Regression_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_10_Logistic_Regression_CheatSheet.tex

REM Session 11: Decision Trees
texify -cp Main_Seminar_MLCoEP_Session_11_Decision_Trees_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_11_Decision_Trees_CheatSheet.tex

REM Session 12: Ensemble RF
texify -cp Main_Seminar_MLCoEP_Session_12_Ensemble_RF_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_12_Ensemble_RF_CheatSheet.tex

REM Session 13: SVM
texify -cp Main_Seminar_MLCoEP_Session_13_SVM_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_13_SVM_CheatSheet.tex

REM Session 14: Naive Bayes
texify -cp Main_Seminar_MLCoEP_Session_14_Naive_Bayes_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_14_Naive_Bayes_CheatSheet.tex

REM Session 15: KNN
texify -cp Main_Seminar_MLCoEP_Session_15_KNN_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_15_KNN_CheatSheet.tex

REM Session 16: KMeans
texify -cp Main_Seminar_MLCoEP_Session_16_KMeans_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_16_KMeans_CheatSheet.tex

REM Session 17: PCA
texify -cp Main_Seminar_MLCoEP_Session_17_PCA_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_17_PCA_CheatSheet.tex

REM Session 18: Titanic Capstone
texify -cp Main_Seminar_MLCoEP_Session_18_Titanic_Capstone_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_18_Titanic_Capstone_CheatSheet.tex

REM Session 19: MLOps Deployment
texify -cp Main_Seminar_MLCoEP_Session_19_MLOps_Deployment_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_19_MLOps_Deployment_CheatSheet.tex

REM Session 20: ME Apps
texify -cp Main_Seminar_MLCoEP_Session_20_ME_Apps_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_20_ME_Apps_CheatSheet.tex

echo.
echo All 20 sessions compiled. Check individual .log files for errors.
pause
