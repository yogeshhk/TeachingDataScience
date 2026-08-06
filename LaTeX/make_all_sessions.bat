@echo off
REM ML CoEP: Compile all 19 sessions (38 driver files: 19 Presentation + 19 CheatSheet)
REM Usage: make_all_sessions.bat
REM Each texify call compiles independently; failures in one do not stop the rest.

echo Compiling all 19 ML CoEP sessions...
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

REM Session 8: Linear Regression
texify -cp Main_Seminar_MLCoEP_Session_8_Linear_Regression_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_8_Linear_Regression_CheatSheet.tex

REM Session 9: Logistic Regression
texify -cp Main_Seminar_MLCoEP_Session_9_Logistic_Regression_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_9_Logistic_Regression_CheatSheet.tex

REM Session 10: Decision Trees
texify -cp Main_Seminar_MLCoEP_Session_10_Decision_Trees_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_10_Decision_Trees_CheatSheet.tex

REM Session 11: Ensemble RF
texify -cp Main_Seminar_MLCoEP_Session_11_Ensemble_RF_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_11_Ensemble_RF_CheatSheet.tex

REM Session 12: SVM
texify -cp Main_Seminar_MLCoEP_Session_12_SVM_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_12_SVM_CheatSheet.tex

REM Session 13: Naive Bayes
texify -cp Main_Seminar_MLCoEP_Session_13_Naive_Bayes_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_13_Naive_Bayes_CheatSheet.tex

REM Session 14: KNN
texify -cp Main_Seminar_MLCoEP_Session_14_KNN_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_14_KNN_CheatSheet.tex

REM Session 15: KMeans
texify -cp Main_Seminar_MLCoEP_Session_15_KMeans_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_15_KMeans_CheatSheet.tex

REM Session 16: PCA
texify -cp Main_Seminar_MLCoEP_Session_16_PCA_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_16_PCA_CheatSheet.tex

REM Session 17: Titanic Capstone
texify -cp Main_Seminar_MLCoEP_Session_17_Titanic_Capstone_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_17_Titanic_Capstone_CheatSheet.tex

REM Session 18: MLOps Deployment
texify -cp Main_Seminar_MLCoEP_Session_18_MLOps_Deployment_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_18_MLOps_Deployment_CheatSheet.tex

REM Session 19: ME Apps
texify -cp Main_Seminar_MLCoEP_Session_19_ME_Apps_Presentation.tex
texify -cp Main_Seminar_MLCoEP_Session_19_ME_Apps_CheatSheet.tex

echo.
echo All 19 sessions compiled. Check individual .log files for errors.
pause
