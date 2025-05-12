# https://colab.research.google.com/drive/1eiUBpmQ4m7Lbxqi2xth1jBaL61XTKdxp?usp=sharing#scrollTo=e9HD5KW3QGFS
# https://www.youtube.com/watch?v=OIenNRt2bjg
import torch
import torch.nn as nn

# Linear regression
# f = w * x 
# here : f = 2 * x

# 0) Training samples, watch the shape!
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_samples = {n_samples}, n_features = {n_features}')

# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# 1) Design Model, the model has to implement the forward pass!

# Here we could simply use a built-in model from PyTorch
# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self,x):
        return self.linear_layer(x)
    
input_size, output_size = n_features, n_features

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f({X_test.item()}) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    y_predicted = model(X)
    l = loss(Y, y_predicted)
    l.backward() # gradiants have got stored in Weights variables in Linear Layer ie parameters
    
    optimizer.step() # update the parameters
    
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        w, b = model.parameters() # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l.item())

print(f'Prediction after training: f({X_test.item()}) = {model(X_test).item():.3f}')