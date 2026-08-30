"""
GNN Graph Classification: MUTAG Mutagenicity Dataset
Classifies molecules as mutagenic or not using GCN and GraphConv architectures.
"""
import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool

os.makedirs('data/TUDataset', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


class GNN(torch.nn.Module):
    """GraphConv-based model: supports edge weights unlike GCNConv."""
    def __init__(self, num_node_features, num_classes, hidden_channels=64):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def run_model(model, name, train_loader, test_loader, epochs=200):
    print(f'\n--- {name} ---')
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, criterion)
        if epoch % 50 == 0 or epoch == 1:
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            print(f'  Epoch {epoch:03d}: Loss={loss:.4f} | Train={train_acc:.4f} Test={test_acc:.4f}')

    final_test = test(model, test_loader)
    print(f'  Final Test Accuracy: {final_test:.4f}')
    return final_test


if __name__ == '__main__':
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    print(f'Dataset: MUTAG')
    print(f'  Graphs: {len(dataset)}, Features: {dataset.num_features}, Classes: {dataset.num_classes}')

    data = dataset[0]
    print(f'  First graph - Nodes: {data.num_nodes}, Edges: {data.num_edges}')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    print(f'  Train: {len(train_dataset)}, Test: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    gcn = GCN(num_features, num_classes, hidden_channels=64)
    run_model(gcn, 'GCN (GCNConv)', train_loader, test_loader, epochs=170)

    gnn = GNN(num_features, num_classes, hidden_channels=64)
    run_model(gnn, 'GNN (GraphConv)', train_loader, test_loader, epochs=200)

    print('\nDone.')
