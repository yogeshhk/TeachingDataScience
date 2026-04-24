"""
GNN Scaling: Mini-batch Training with ClusterData on PubMed
Demonstrates handling large graphs using cluster-based stochastic partitioning.
"""
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv

os.makedirs('data/Planetoid', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_nodes = 0
    for sub_data in train_loader:
        sub_data = sub_data.to(DEVICE)
        optimizer.zero_grad()
        out = model(sub_data.x, sub_data.edge_index)
        loss = criterion(out[sub_data.train_mask], sub_data.y[sub_data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sub_data.train_mask.sum().item()
        total_nodes += sub_data.train_mask.sum().item()
    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    data = data.to(DEVICE)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = {}
    for split, mask in [('train', data.train_mask), ('val', data.val_mask), ('test', data.test_mask)]:
        correct = pred[mask] == data.y[mask]
        accs[split] = int(correct.sum()) / int(mask.sum())
    return accs


if __name__ == '__main__':
    dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
    data = dataset[0]

    print(f'Dataset: PubMed (Planetoid)')
    print(f'  Nodes: {data.num_nodes}, Edges: {data.num_edges}')
    print(f'  Features: {dataset.num_features}, Classes: {dataset.num_classes}')
    print(f'  Training nodes: {data.train_mask.sum()} ({int(data.train_mask.sum()) / data.num_nodes:.3%})')

    # Create cluster subgraphs for mini-batch training
    print('\nPartitioning graph into 128 clusters...')
    torch.manual_seed(12345)
    cluster_data = ClusterData(data, num_parts=128)
    train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)

    print(f'Cluster batches: {len(train_loader)}')
    for step, sub_data in enumerate(train_loader):
        print(f'  Batch {step + 1}: {sub_data.num_nodes} nodes')
        if step >= 3:
            print('  ...')
            break

    model = GCN(dataset.num_features, dataset.num_classes, hidden_channels=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print('\nTraining GCN with cluster mini-batches...')
    for epoch in range(1, 51):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        if epoch % 10 == 0:
            accs = evaluate(model, data)
            print(f'  Epoch {epoch:02d}: Loss={loss:.4f} | '
                  f'Train={accs["train"]:.4f} Val={accs["val"]:.4f} Test={accs["test"]:.4f}')

    accs = evaluate(model, data)
    print(f'\nFinal Test Accuracy: {accs["test"]:.4f}')
    print('Done.')
