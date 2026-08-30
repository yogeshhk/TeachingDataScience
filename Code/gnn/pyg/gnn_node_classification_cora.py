"""
GNN Node Classification: Cora Citation Network
Compares MLP, GCN, and GAT on transductive node classification (Cora dataset).
"""
import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import Linear
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GATConv
from sklearn.manifold import TSNE

os.makedirs('data/Planetoid', exist_ok=True)
os.makedirs('plots', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def visualize_tsne(h, color, title='Embeddings', filename='plots/cora_tsne.png'):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.cpu().numpy(), cmap='Set2')
    plt.title(title)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index=None):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=8, heads=8):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index) if hasattr(model, 'conv1') else model(data.x)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index) if hasattr(model, 'conv1') else model(data.x)
    pred = out.argmax(dim=1)
    results = {}
    for split, mask in [('train', data.train_mask), ('val', data.val_mask), ('test', data.test_mask)]:
        correct = pred[mask] == data.y[mask]
        results[split] = int(correct.sum()) / int(mask.sum())
    return results, out


def run_experiment(model, data, name, lr=0.01, weight_decay=5e-4, epochs=200):
    print(f'\n--- {name} ---')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion)
        if epoch % 50 == 0:
            results, _ = evaluate(model, data)
            print(f'  Epoch {epoch:03d}: Loss={loss:.4f} | '
                  f'Train={results["train"]:.4f} Val={results["val"]:.4f} Test={results["test"]:.4f}')

    results, out = evaluate(model, data)
    print(f'  Final Test Accuracy: {results["test"]:.4f}')
    visualize_tsne(out, data.y, title=f'{name} Embeddings', filename=f'plots/cora_tsne_{name.lower()}.png')
    return results


if __name__ == '__main__':
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    print(f'Dataset: Cora (Planetoid)')
    print(f'  Nodes: {data.num_nodes}, Edges: {data.num_edges}')
    print(f'  Features: {dataset.num_features}, Classes: {dataset.num_classes}')
    print(f'  Training nodes: {data.train_mask.sum()} ({int(data.train_mask.sum()) / data.num_nodes:.0%})')

    data = data.to(DEVICE)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # MLP baseline (no graph structure)
    mlp = MLP(num_features, num_classes, hidden_channels=16).to(DEVICE)
    run_experiment(mlp, data, 'MLP', lr=0.01, weight_decay=5e-4, epochs=200)

    # GCN (graph convolutional network)
    gcn = GCN(num_features, num_classes, hidden_channels=16).to(DEVICE)
    run_experiment(gcn, data, 'GCN', lr=0.01, weight_decay=5e-4, epochs=200)

    # GAT (graph attention network)
    gat = GAT(num_features, num_classes, hidden_channels=8, heads=8).to(DEVICE)
    run_experiment(gat, data, 'GAT', lr=0.005, weight_decay=5e-4, epochs=200)

    print('\nDone. Plots saved to plots/cora_tsne_*.png')
