# https://www.youtube.com/watch?v=3Kb0QS6z7WA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4

# Chain rule
# x -> a(x) -y-> b(y) -z->
# dz/dx = dz/dy . dy/dx

# In Linear regression:
# Loss = (y' - y)^2 = (wx-y)^2
# need dLoss/dw, so make w.required_grad = True
# ![images/04_backpropogation] [04_backpropogation_example.png]
# No need derivatives of dLoss/dx and dLoss/dy as they are training inputs and constants
import torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass
y_hat = w * x
loss = (y_hat - y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)

# update weights, then fwd pass , then backword pass

