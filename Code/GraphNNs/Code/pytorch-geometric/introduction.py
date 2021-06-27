# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data1 = Data(x=x, edge_index=edge_index)

print(data1)

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data2 = Data(x=x, edge_index=edge_index.t().contiguous())
print(data2)
