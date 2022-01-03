# import torch
# from torch_geometric.data import Data
#
# edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long)
# x = torch.tensor([[-1],[0],[1]],dtype=torch.float)
#
# data = Data(x=x,edge_index=edge_index)
#
# print(data) # edge index : [2, num_edges], x = [num_nodes, num_node_features]
#
# print("Fields: ", data.keys)
#
# print("x: ", data['x'])
#
# print("Num nodes: ", data.num_nodes)
#
# print("Num Edges: ", data.num_edges)
#
# print("Num node features: ", data.num_node_features)
#
# print("Has Isolated Nodes: ", data.has_isolated_nodes())
#
# print("Has Self loops: ", data.has_self_loops())
#
# print("Is directed: ", data.is_directed())

# from torch_geometric.datasets import TUDataset
# from torch_geometric.loader import DataLoader
# from torch_scatter import scatter_mean
#
# dataset = TUDataset(root='data/ENZYMES', name='ENZYMES', use_node_attr=True)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# torch_geometric.data.Batch inherits from torch_geometric.data.Data and contains an additional attribute called 'batch'
# for batch in loader:
#     x = scatter_mean(batch.x, batch.batch, dim=0)
#     print(x.size()) # y == 32, as its same as  num_graphs in the batch
#

# from torch_geometric.datasets import ShapeNet
# import torch_geometric.transforms as T
# convert point cloud dataset into graph dataset by generating nearest neighbor graphs from point clouds via transforms
# dataset = ShapeNet(root='data/ShapeNet', categories=['Airplane'], pre_transform=T.KNNGraph(k=6))
# print(dataset[0])

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='data/Cora', name='Cora')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x,dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct)/int(data.test_mask.sum())
print(f"Accuracy: {acc:.4f}")


