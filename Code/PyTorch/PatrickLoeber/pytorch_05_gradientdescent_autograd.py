# https://www.youtube.com/watch?v=E-I2DNVzQLg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=5

# Step 2
#   - Prediction : Manual
#   - Gradient Computation : Autograd
#   - Loss Computation : Manual
#   - Parameter Update : Manual


import torch

X = torch.tensor([1,2,3,4],dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# Prediction
def forward(x):
    return w * x

def loss(y,y_predicted):
    return ((y_predicted - y)**2).mean()


# prediction before training for x = 5
print(f"prediction before training : {forward(5)}")

# Training
learning_rate = 0.01
n_iters = 50
for epoch in range(n_iters):
    # prediction
    y_pred = forward(X)
    l = loss(Y, y_pred)

    # gradient == backpropogation
    l.backward() # dl/dw
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()

    print(f"epoch {epoch+1}, w = {w}, loss = {l}")

print(f"prediction after training: {forward(5)}")

