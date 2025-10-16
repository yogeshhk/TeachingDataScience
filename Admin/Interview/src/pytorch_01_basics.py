# https://jovian.ai/aakashns/01-pytorch-basics

import torch
t1 = torch.tensor(4.)
t2 = torch.tensor([1.,2,3,4])
t3 = torch.tensor([[5,6],[3,.4],[1,2]])
t4 = torch.tensor([
    [[11, 12, 13],
     [13, 14, 15]],
    [[15, 16, 17],
     [17, 18, 19.]]])

x = torch.tensor(3.)
w = torch.tensor(4.,requires_grad=True)
b = torch.tensor(5.,requires_grad=True)
y = w * x + b
y.backward() # derivative wrt each input variable is calculated and stored in respective variables
print(w.grad,b.grad)

import numpy as np
x = np.array([[1,2],[3,4.]])
y = torch.from_numpy(x)
z = y.numpy()
print(z)