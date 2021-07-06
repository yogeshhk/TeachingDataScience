# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=2

import torch
x = torch.empty(2,3, 1)
print(x)

x = torch.rand(2,2)
print(x)

x = torch.zeros(3)
print(x)

x = torch.ones(2)
print(x)

x = torch.ones(2,2,dtype=torch.float)
print(type(x))
print(type(x.data))
print(x.data)
print(x.size())
print(x.dtype)
print(x.grad_fn)

x = torch.tensor([2.5,0.1])
print(x)

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x + y # element wise
print(z)

z = torch.add(x,y) # element wise
print(z)

y.add_(x) #in place
print(y)

z = x * y # element wise
print(z)

z = torch.matmul(x,y) # element wise
print(z)

# Slicing
x = torch.rand(5,3)
print(x)
print(x[:,1])
print(x[1,:])
print(x[1,1])

x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y)

y = x.view(-1,8) # for -1 it automatically calculates
print(y)
print(y.size())

import numpy as np
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print(type(b))
# b an d a share same  memory if both on CPU
# modifying one will change other
a.add_(1)
print(a)
print(b)

a = np.ones(5) # only on CPU
print(a)
b = torch.from_numpy(a) # only to tensor on CPU
print(b)

x = torch.ones(5,requires_grad=True) # default is False
print(x)



