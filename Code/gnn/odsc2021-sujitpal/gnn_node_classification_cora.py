"""
GNN Node Classification: Cora Citation Network
Compares GCN, GAT, and GraphSAGE for transductive node classification on Cora.
Source: ODSC West 2021 tutorial by Sujit Pal (solution notebook 02)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

os.makedirs('data/cora', exist_ok=True)
os.makedirs('plots', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_GCN_LAYERS = 3
DROPOUT_PCT = 0.5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-3
NUM_EPOCHS = 500
LOG_EVERY = 100


# ── Data ─────────────────────────────────────────────────────────────────────

dataset = Planetoid(root='data/cora', name='Cora')
print(f'Dataset: Cora | Features: {dataset.num_features} | Classes: {dataset.num_classes}')
data0 = dataset[0]
print(f'  Nodes: {data0.num_nodes} | Edges: {data0.num_edges}')
print(f'  Train/Val/Test nodes: {data0.train_mask.sum()}/{data0.val_mask.sum()}/{data0.test_mask.sum()}')

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
val_loader   = DataLoader(dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

INPUT_DIM  = dataset.num_features
OUTPUT_DIM = dataset.num_classes


# ── Models ────────────────────────────────────────────────────────────────────

class NodeClassifier(nn.Module):
    """Generic node classifier - swap conv_cls to get GCN / GAT / SAGE."""

    def __init__(self, input_dim, num_graph_layers, hidden_dim, output_dim,
                 dropout_pct, conv_cls=GCNConv):
        super().__init__()
        self.num_graph_layers = num_graph_layers
        self.dropout_pct = dropout_pct

        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(input_dim, hidden_dim))
        for _ in range(num_graph_layers - 1):
            self.convs.append(conv_cls(hidden_dim, hidden_dim))

        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_pct),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_pct, training=self.training)
        x = self.clf_head(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


# ── Training helpers ─────────────────────────────────────────────────────────

def train_step(model, optimizer, loader):
    model.train()
    total_rows = total_loss = total_correct = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred  = model(batch)[batch.train_mask]
        label = batch.y[batch.train_mask]
        loss  = model.loss(pred, label)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        total_correct += pred.argmax(dim=1).eq(label).sum().item()
        total_rows    += batch.train_mask.sum().item()
    return total_loss / total_rows, total_correct / total_rows


def eval_step(model, loader, use_val=False):
    model.eval()
    total_rows = total_loss = total_correct = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        mask  = batch.val_mask if use_val else batch.test_mask
        with torch.no_grad():
            pred  = model(batch)[mask]
            label = batch.y[mask]
            loss  = model.loss(pred, label)
            total_loss    += loss.item()
            total_correct += pred.argmax(dim=1).eq(label).sum().item()
            total_rows    += mask.sum().item()
    return total_loss / total_rows, total_correct / total_rows


def train_loop(model, optimizer, num_epochs=NUM_EPOCHS):
    history = []
    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_step(model, optimizer, train_loader)
        va_loss, va_acc = eval_step(model, val_loader, use_val=True)
        history.append((tr_loss, tr_acc, va_loss, va_acc))
        if epoch == 0 or (epoch + 1) % LOG_EVERY == 0:
            print(f'  Epoch {epoch+1:4d}: Train loss={tr_loss:.4f} acc={tr_acc:.4f} | '
                  f'Val loss={va_loss:.4f} acc={va_acc:.4f}')
    return history


def save_plots(history, name):
    tr_losses, tr_accs, va_losses, va_accs = zip(*history)
    xs = np.arange(len(tr_losses))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(xs, tr_losses, label='train'); ax1.plot(xs, va_losses, label='val')
    ax1.set_ylabel('loss'); ax1.legend()
    ax2.plot(xs, tr_accs, label='train'); ax2.plot(xs, va_accs, label='val')
    ax2.set_ylabel('accuracy'); ax2.legend()
    fig.suptitle(name)
    path = f'plots/node_class_{name.lower()}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Run experiments ───────────────────────────────────────────────────────────

def run(name, conv_cls):
    print(f'\n=== {name} ===')
    model = NodeClassifier(INPUT_DIM, NUM_GCN_LAYERS, HIDDEN_DIM, OUTPUT_DIM,
                           DROPOUT_PCT, conv_cls=conv_cls).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = train_loop(model, optimizer)
    save_plots(history, name)
    _, test_acc = eval_step(model, test_loader, use_val=False)
    print(f'  Test accuracy: {test_acc:.5f}')
    return test_acc


if __name__ == '__main__':
    gcn_acc  = run('GCN',       GCNConv)
    gat_acc  = run('GAT',       GATConv)
    sage_acc = run('GraphSAGE', SAGEConv)

    print('\n=== Summary ===')
    print(f'  GCN       test acc: {gcn_acc:.5f}')
    print(f'  GAT       test acc: {gat_acc:.5f}')
    print(f'  GraphSAGE test acc: {sage_acc:.5f}')
    print('Done. Plots saved to plots/node_class_*.png')
