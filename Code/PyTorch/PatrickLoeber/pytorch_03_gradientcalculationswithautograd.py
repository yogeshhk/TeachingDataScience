# https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3

import torch
x = torch.randn(3, requires_grad=True)
print(x)

# Finding gradient of some function wrt 'x' meaning wrt [x1,x2,x3]
# this will be partial derivative as 'x' is vector with dimension > 1

# Whenever, using 'x' an operation is created then pytorch will create
# a computation graph

y = x + 2

# Now computation graph will look like a perceptron with 'x' and '2' as input neurons,
# '+' in the node and 'y' as the output nueron
# Now, its possible to calculate gradient dy/dx, for back propogation
# y.grad_fn called AddBackword0 will be dy/dx
print(y)

z = y * y *2
print(z)  # now z.grad_fn is "MulBackword0"

z = z.mean() # now z.grad_fn is "MeanBackword0"
print(z)

z  = z + 3
print(z) # now z.grad_fn is "AddBackword0"

# to compute the actual gradient value, the actual value gets stored in x.grad
z.backward() # due to 'mean' z is a scalar, so no need to pass anything here
print(x.grad)

z = y * y *2
print(z)
# z.backward() wont work
# We need to pass the point at which dz/dx needs to be calculated
# The tensor passed into the backward function acts like weights
# for a weighted output of gradient. Mathematically,
# this is the vector multiplied by the Jacobian matrix of non-scalar tensors
v = torch.tensor([0.1,1.0,0.001], dtype=torch.float)
z.backward(v)
print(x.grad)


# Autograd records a graph of all the operations performed on a
# gradient enabled tensor and creates an acyclic graph called the
# dynamic computational graph. The leaves of this graph are input tensors
# and the roots are output tensors.
# Gradients are calculated by tracing the graph from the root to the leaf
# and multiplying every gradient in the way using the chain rule.

# For some situations where you dont want autograd to add some operations to graph
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():
#   ... do the ops

# Notes: gradients get accumulated

weights = torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
# tensor([3., 3., 3., 3.])
# tensor([6., 6., 6., 6.])

# to avoid accumulation call weights.grad.zero_()







