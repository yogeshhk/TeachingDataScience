# https://www.youtube.com/watch?v=OGpQxIkR4ao&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=8

# generic steps
#   1. Design model (input size, output size, forward pass)
#   2. Construct loss and optimizer
#   3. Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
print(f"Samples {n_samples}, Features {n_features}")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=7)

# Good to scale for logistic regression
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# y is currently one rwo, need to make it column
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)


# design model
class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_input_features,1)
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

num_samples, n_features = X_train.shape

model = LogisticRegression(n_features)


# define loss and optimizer
criterion = nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# training loop
n_iters = 100
for epoch in range(n_iters):
    y_pred = model(X_train)
    loss = criterion(y_pred,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch %10 == 0:
        print(f"epoch {epoch+1} loss {loss.item()}")

# Plot
with torch.no_grad():
    predicted = model(X_test)
    y_predicted_labels = predicted.round()
    acc = y_predicted_labels.eq(y_test).sum()/y_test.shape[0]
    print(f"Accuracy {acc}")

