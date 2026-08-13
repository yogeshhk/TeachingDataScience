# https://jovian.ai/aakashns/02-linear-regression

import numpy as np
import torch
# # Input (temp, rainfall, humidity)
# inputs = np.array([[73, 67, 43],
#                    [91, 88, 64],
#                    [87, 134, 58],
#                    [102, 43, 37],
#                    [69, 96, 70]], dtype='float32')
#
# # Targets (apples, oranges)
# targets = np.array([[56, 70],
#                     [81, 101],
#                     [119, 133],
#                     [22, 37],
#                     [103, 119]], dtype='float32')
#
# # Convert inputs and targets to tensors
# inputs = torch.from_numpy(inputs)
# targets = torch.from_numpy(targets)
#
# # Weights and biases
# w = torch.randn(2, 3, requires_grad=True)
# b = torch.randn(2, requires_grad=True)
# print(w)
# print(b)
#
# def model(x):
#     return x @ w.t() + b
#
# # MSE loss
# # .numel method of a tensor returns the number of elements in a tensor.
# def mse(t1, t2):
#     diff = t1 - t2
#     return torch.sum(diff * diff) / diff.numel()
#
# # Train for 100 epochs
# for i in range(200):
#     preds = model(inputs)
#     loss = mse(preds, targets)
#     loss.backward()
#     with torch.no_grad():
#         w -= w.grad * 1e-5
#         b -= b.grad * 1e-5
#         w.grad.zero_()
#         b.grad.zero_()
#
# # Calculate loss
# preds = model(inputs)
# loss = mse(preds, targets)
# print(loss)
# print(preds)
# print(targets)

from torch.utils.data import TensorDataset
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Define dataset
train_ds = TensorDataset(inputs, targets)

from torch.utils.data import DataLoader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = torch.nn.Linear(3, 2)
preds = model(inputs)

import torch.nn.functional as F
loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:
            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

fit(100, model, loss_fn, opt, train_dl)
preds = model(inputs)
print(preds)
print(targets)
print(model(torch.tensor([[75, 63, 44.]])))