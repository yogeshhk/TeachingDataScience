# https://www.youtube.com/watch?v=VVDHU_TWwUg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=6

# generic steps
#   1. Design model (input, output size, forward pass)
#   2. Construct loss and optimizer
#   3. Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights


# Step 3/4
#   - Prediction : Manual
#   - Gradient Computation : Autograd
#   - Loss Computation : Manual
#   - Parameter Update : Manual


import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

# w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# # Prediction
# def forward(x):
#     return w * x
n_samples, n_features = X.shape
print(n_samples,n_features)
input_size = n_features
output_size = n_features

# model = nn.Linear(input_size,output_size)

class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.lin = nn.Linear(input_dim,output_dim)
    def forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)

# def loss(y,y_predicted):
#     return ((y_predicted - y)**2).mean()

# Training
learning_rate = 0.01

loss = nn.MSELoss() # callable function
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# prediction before training for x = 5
X_test = torch.tensor([5], dtype=torch.float32)
print(f"prediction before training : {model(X_test).item()}")

n_iters = 1000
for epoch in range(n_iters):
    # prediction
    y_pred = model(X)
    l = loss(Y, y_pred)

    # gradient == backpropogation
    l.backward() # dl/dw

    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()

    # w.grad.zero_()
    optimizer.zero_grad()
    [w,b] = model.parameters()
    print(f"epoch {epoch+1}, w = {w[0][0].item()}, loss = {l}")

print(f"prediction after training: {model(X_test).item()}")

