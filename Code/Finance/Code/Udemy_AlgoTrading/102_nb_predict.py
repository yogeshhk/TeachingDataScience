# -*- coding: utf-8 -*-
"""
Using trained Naive Bayes classifier to predict sentiment of a news article

@author: Mayank Rasu
"""
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

#############Importing trained classifier and fitted vectorizer################
nb_clf = pickle.load(open("nb_clf_crude_oil", 'rb'))
vectorizer = pickle.load(open("vectorizer_crude_oil", 'rb'))

##############Predict sentiment using the trained classifier###################

# Import test data set
data_pred = pd.read_csv("CrudeOil_News_Articles_test.csv", encoding = "ISO-8859-1")
X_test = data_pred.iloc[:,1] # extract column with news articl
X_vec_test = vectorizer.transform(X_test) #don't use fit_transform here because the model is already fitted
X_vec_test = X_vec_test.todense() #convert sparse matrix to dense

# Transform data by applying term frequency inverse document frequency (TF-IDF) 
tfidf = TfidfTransformer() #by default applies "l2" normalization
X_tfidf_test = tfidf.fit_transform(X_vec_test)
X_tfidf_test = X_tfidf_test.todense()


# Predict the sentiment values
y_pred = nb_clf.predict(X_tfidf_test)