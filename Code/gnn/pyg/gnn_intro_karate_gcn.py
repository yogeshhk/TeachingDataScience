"""
GNN Introduction: KarateClub Dataset with GCN
Teaches basic PyG data handling and GCN message-passing on Zachary's karate club graph.
"""
import os
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx

os.makedirs('plots', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def visualize_graph(G, color, filename='plots/karate_graph.png'):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap='Set2')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


def visualize_embedding(h, color, epoch=None, loss=None, filename=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color.cpu().numpy(), cmap='Set2')
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    fname = filename or f'plots/karate_embedding_epoch{epoch:03d}.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h


def train(model, data, optimizer, criterion):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


if __name__ == '__main__':
    dataset = KarateClub()
    print(f'Dataset: {dataset}')
    print(f'  Graphs: {len(dataset)}, Features: {dataset.num_features}, Classes: {dataset.num_classes}')

    data = dataset[0]
    print(f'\nGraph info:')
    print(f'  Nodes: {data.num_nodes}, Edges: {data.num_edges}')
    print(f'  Avg degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'  Training nodes: {data.train_mask.sum()} ({int(data.train_mask.sum()) / data.num_nodes:.0%})')

    # Visualize graph structure
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y, filename='plots/karate_graph.png')

    # Move data to device
    data = data.to(DEVICE)

    model = GCN(dataset.num_features, dataset.num_classes).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print('\nTraining GCN on KarateClub...')
    for epoch in range(401):
        loss, h = train(model, data, optimizer, criterion)
        if epoch % 50 == 0:
            print(f'  Epoch {epoch:03d}: Loss = {loss.item():.4f}')
            visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)

    print('\nDone. Plots saved to plots/karate_*.png')
