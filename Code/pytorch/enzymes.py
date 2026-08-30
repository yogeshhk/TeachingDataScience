# Dmitry Korobchenko - PyTorch Geometric for Graph Neural Nets | PyData Yerevan
# https://www.youtube.com/watch?v=knmPoaqCoyw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.optim as optim

dataset = TUDataset(root="../data", name="ENZYMES")
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)


class GNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


model = GNN(input_dim=3, num_classes=6)
optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()
model.train()

for epoch in range(2):
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index,data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f" Epoch {epoch}, loss {loss}")


new_x = dataset[2]
output = model(new_x.x, new_x.edge_index, new_x.batch)
print(torch.argmax(output))