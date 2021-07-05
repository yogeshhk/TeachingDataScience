# https://www.youtube.com/watch?v=YAJ5XBwlN4o&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=7

# generic steps
#   1. Design model (input, output size, forward pass)
#   2. Construct loss and optimizer
#   3. Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100,
                                            n_features=1,
                                            noise=20,
                                            random_state=7)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)
num_samples, n_features = X.shape

# design model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# define loss and optimizer
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# training loop
n_iters = 100
for epoch in range(n_iters):
    y_pred = model(X)
    loss = criterion(y_pred,y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch+1 %10 == 0:
        print(f"epoch {epoch+1} loss {loss.item()}")
# Plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()

