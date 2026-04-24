"""
GNN Explanation: Mutagenicity Dataset with Captum Attribution
Trains a GNN for molecule mutagenicity classification then explains predictions
using Saliency and Integrated Gradients (Captum library).
"""
import os
import random
from collections import defaultdict
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_add_pool
from torch_geometric.utils import to_networkx
from captum.attr import Saliency, IntegratedGradients

os.makedirs('data/TUDataset', exist_ok=True)
os.makedirs('plots', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

ATOM_MAP = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']


def to_molecule(data):
    g = to_networkx(data, node_attrs=['x'])
    for u, d in g.nodes(data=True):
        d['name'] = ATOM_MAP[d['x'].index(1.0)]
        del d['x']
    return g


def draw_molecule(g, edge_mask=None, title='', filename='plots/molecule.png'):
    g = g.copy().to_undirected()
    node_labels = {u: d['name'] for u, d in g.nodes(data=True)}
    pos = nx.spring_layout(g, pos=nx.planar_layout(g))
    edge_color = 'black' if edge_mask is None else [edge_mask.get((u, v), 0.0) for u, v in g.edges()]
    widths = None if edge_mask is None else [x * 10 for x in edge_color]
    plt.figure(figsize=(8, 4))
    plt.title(title)
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues, node_color='azure')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, dim=32):
        super().__init__()
        self.dim = dim
        self.conv1 = GraphConv(num_features, dim)
        self.conv2 = GraphConv(dim, dim)
        self.conv3 = GraphConv(dim, dim)
        self.conv4 = GraphConv(dim, dim)
        self.conv5 = GraphConv(dim, dim)
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight).relu()
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


def train(model, loader, optimizer, epoch):
    model.train()
    if epoch == 51:
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.5
    loss_all = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def model_forward(edge_mask, data):
    # edge_mask may be moved to CPU by Captum's internal batching; always re-send to DEVICE
    edge_mask = edge_mask.to(DEVICE)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(DEVICE)
    return model(data.x.to(DEVICE), data.edge_index.to(DEVICE), batch, edge_mask)


def explain(method, data, target=0):
    input_mask = torch.ones(data.edge_index.shape[1], requires_grad=True).to(DEVICE)
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data,),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        sal = Saliency(model_forward)
        mask = sal.attribute(input_mask, target=target, additional_forward_args=(data,))
    else:
        raise ValueError(f'Unknown method: {method}')
    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in zip(edge_mask, *data.edge_index):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


if __name__ == '__main__':
    random.seed(42)
    dataset = TUDataset('data/TUDataset', name='Mutagenicity').shuffle()
    n_test = len(dataset) // 10
    test_dataset = dataset[:n_test]
    train_dataset = dataset[n_test:]

    print(f'Dataset: Mutagenicity')
    print(f'  Total: {len(dataset)}, Train: {len(train_dataset)}, Test: {len(test_dataset)}')
    print(f'  Features: {dataset.num_features}, Classes: {dataset.num_classes}')

    train_loader = DataLoader(train_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = Net(dataset.num_features, dataset.num_classes, dim=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('\nTraining GNN on Mutagenicity...')
    for epoch in range(1, 101):
        loss = train(model, train_loader, optimizer, epoch)
        if epoch % 20 == 0:
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            print(f'  Epoch {epoch:03d}: Loss={loss:.4f} | Train={train_acc:.4f} Test={test_acc:.4f}')

    final_acc = test(model, test_loader)
    print(f'Final Test Accuracy: {final_acc:.4f}')

    # Explain predictions on a non-mutagenic molecule
    model.eval()
    non_mutagen = [t for t in test_dataset if not t.y.item()]
    if non_mutagen:
        sample = random.choice(non_mutagen)
        sample = sample.to(DEVICE)
        mol = to_molecule(sample.cpu())

        draw_molecule(mol, title='Molecule (no explanation)', filename='plots/explanation_molecule.png')

        for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
            edge_mask = explain(method, sample, target=0)
            edge_mask_dict = aggregate_edge_directions(edge_mask, sample.cpu())
            draw_molecule(mol, edge_mask=edge_mask_dict, title=title,
                          filename=f'plots/explanation_{method}.png')

    print('\nDone. Plots saved to plots/explanation_*.png')
