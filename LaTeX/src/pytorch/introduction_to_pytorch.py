# Introduction to PyTorch - PyTorch Youtube Channel https://www.youtube.com/watch?v=IC0_FRiX-sw
import torch
z = torch.zeros(5,3)
# print(z)
# print(z.dtype)
i = torch.ones((5,3), dtype=torch.int8)
# print(i)

torch.manual_seed(1729)
print(torch.rand(2,2))

print(torch.rand(2,2))

torch.manual_seed(1729)
print(torch.rand(2,2))
