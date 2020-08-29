"""
    Ref: https://www.analyticsvidhya.com/blog/2020/05/pycaret-machine-learning-model-seconds/
"""
import pandas as pd
from pycaret import classification

data_classification = pd.read_csv('data/loan_train_data.csv')
print(data_classification.head())

"""
In this step, PyCaret performs some basic preprocessing tasks, like ignoring the IDs and Date Columns, 
imputing the missing values, encoding the categorical variables, and splitting the dataset into the train-test
split for the rest of the modeling steps. When you run the setup function, it will first confirm the data types, 
and then if you press enter, it will create the environment for you to go ahead
"""
classification_setup = classification.setup(data=data_classification, target='Personal Loan')

"""
train a decision tree model for which we have to pass “dt” and it will return a table with k-fold cross-validated 
scores of common evaluation metrics used for classification models.
"""
classification_dt = classification.create_model('dt') # another option: “xgboost“
#
# """"
# we can define the number of folds using the fold parameter within the tune_model function. Or we can change the number
# of iterations using the n_iter parameter. Increasing the n_iter parameter will obviously increase the training time
# but will give a much better performance.
# """
# tune_catboost = classification.tune_model('catboost')
#
# """
# Let’s train a boosting ensemble model here. It will also return a table with k-fold cross-validated scores
# of common evaluation metrics:
# """
# boosting = classification.ensemble_model(classification_dt, method= 'Boosting')
#
# # compare performance of different classification models
# classification.compare_models()
#
# classification.plot_model(classification_dt, plot = 'auc') # AUC-ROC plot
# classification.plot_model(classification_dt, plot = 'boundary') # Decision Boundary
# classification.plot_model(classification_dt, plot = 'pr') # Precision Recall Curve
# classification.plot_model(classification_dt, plot = 'vc') # Validation Curve
#
# classification.evaluate_model(classification_dt)
#
# # make predictions
# test_data_classification = pd.read_csv('data/loan_test_data.csv')
# predictions = classification.predict_model(classification_dt, data=test_data_classification)
# print(predictions)
