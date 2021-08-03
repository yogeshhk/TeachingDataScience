"""
Beating the benchmark for Liberty Mutual Fund @ Kaggle

__author__ : Abhishek Thakur
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

tr = train[['var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var17']]
ts = test[['var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var17']]

tr = np.nan_to_num(np.array(tr))
ts = np.nan_to_num(np.array(ts))

clf = Ridge()
clf.fit(tr, train['target'].values)
preds = clf.predict(ts)

sample['target'] = preds

sample.to_csv('submission.csv', index = False)


