"""
PyTorch Basics: Linear Regression
Demonstrates gradient descent and parameter learning on a simple f = 2*x problem.
"""
import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'Samples: {n_samples}, Features: {n_features}')

X_test = torch.tensor([5], dtype=torch.float32)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression(n_features, n_features)
print(f'Before training: f({X_test.item()}) = {model(X_test).item():.3f}')

lr = 0.01
n_epochs = 100
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    y_pred = model(X)
    loss = loss_fn(Y, y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        w, b = model.parameters()
        print(f'Epoch {epoch + 1:3d}: w = {w[0][0].item():.4f}, loss = {loss.item():.6f}')

print(f'After training:  f({X_test.item()}) = {model(X_test).item():.3f}  (expected: 10.0)')
