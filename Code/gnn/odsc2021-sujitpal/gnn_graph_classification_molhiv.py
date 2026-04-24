"""
GNN Graph Classification: ogbg-molhiv HIV Activity Dataset
Compares mean/max/sum global pooling strategies for binary molecule classification.
Uses OGB AtomEncoder for categorical atom features.
Source: ODSC West 2021 tutorial by Sujit Pal (solution notebook 03)
"""
import os
import functools
import torch
# PyTorch 2.6+ changed torch.load default to weights_only=True; OGB's cached
# PyG Data objects contain non-tensor globals that require weights_only=False.
torch.load = functools.partial(torch.load, weights_only=False)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder
from sklearn.metrics import roc_auc_score

os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Hyperparameters
BATCH_SIZE     = 32
HIDDEN_DIM     = 256
NUM_GCN_LAYERS = 5
DROPOUT_PCT    = 0.5
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 30
LOG_EVERY      = 5


# ── Data ─────────────────────────────────────────────────────────────────────

dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data')
print(f'Dataset: ogbg-molhiv | Graphs: {len(dataset)}')
print(f'  Features: {dataset.num_features} | Classes: {dataset.num_classes}')

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx['train']], batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(dataset[split_idx['valid']], batch_size=BATCH_SIZE)
test_loader  = DataLoader(dataset[split_idx['test']],  batch_size=BATCH_SIZE)

INPUT_DIM  = dataset.num_features
OUTPUT_DIM = dataset.num_classes


# ── Models ────────────────────────────────────────────────────────────────────

class GraphClassifier(nn.Module):
    """GCN graph classifier with configurable global pooling."""

    def __init__(self, hidden_dim, output_dim, num_graph_layers, dropout_pct,
                 pooling='mean'):
        super().__init__()
        self.num_graph_layers = num_graph_layers
        self.dropout_pct = dropout_pct
        self.pooling = pooling

        self.encoder = AtomEncoder(hidden_dim)   # categorical → continuous

        self.convs = nn.ModuleList([
            pyg_nn.GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_graph_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_graph_layers - 1)
        ])
        self.clf_head = nn.Linear(hidden_dim, output_dim)

        if pooling == 'mean':
            self.pool = pyg_nn.global_mean_pool
        elif pooling == 'max':
            self.pool = pyg_nn.global_max_pool
        elif pooling == 'sum':
            self.pool = pyg_nn.global_add_pool
        else:
            raise ValueError(f'Unknown pooling: {pooling}')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        for i in range(self.num_graph_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_pct, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.pool(x, batch)
        return self.clf_head(x)


# ── Training helpers ─────────────────────────────────────────────────────────

def compute_roc_auc(label_preds):
    y_true  = [lp[0] for lp in label_preds]
    y_score = [lp[1][1] for lp in label_preds]
    return roc_auc_score(y_true, y_score)


def train_step(model, optimizer, loss_fn, loader):
    model.train()
    total_rows = total_loss = total_correct = 0
    label_preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred  = model(batch)
        label = batch.y.squeeze(dim=-1)
        label_cat = F.one_hot(label, num_classes=2).float()
        loss  = loss_fn(pred, label_cat)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        total_correct += pred.argmax(dim=-1).eq(label).sum().item()
        total_rows    += batch.num_graphs
        label_preds.extend(zip(label.detach().cpu().numpy(),
                               pred.detach().cpu().numpy()))
    return total_loss / total_rows, total_correct / total_rows, label_preds


def eval_step(model, loss_fn, loader):
    model.eval()
    total_rows = total_loss = total_correct = 0
    label_preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.no_grad():
            pred  = model(batch)
            label = batch.y.squeeze(dim=-1)
            label_cat = F.one_hot(label, num_classes=2).float()
            loss  = loss_fn(pred, label_cat)
            total_loss    += loss.item()
            total_correct += pred.argmax(dim=-1).eq(label).sum().item()
            total_rows    += batch.num_graphs
            label_preds.extend(zip(label.cpu().numpy(), pred.cpu().numpy()))
    return total_loss / total_rows, total_correct / total_rows, label_preds


def train_loop(model, optimizer, loss_fn, num_epochs=NUM_EPOCHS):
    history = []
    for epoch in range(num_epochs):
        tr_loss, tr_acc, tr_lp = train_step(model, optimizer, loss_fn, train_loader)
        va_loss, va_acc, va_lp = eval_step(model, loss_fn, val_loader)
        tr_auc = compute_roc_auc(tr_lp)
        va_auc = compute_roc_auc(va_lp)
        history.append((tr_loss, tr_acc, tr_auc, va_loss, va_acc, va_auc))
        if epoch == 0 or (epoch + 1) % LOG_EVERY == 0:
            print(f'  Epoch {epoch+1:3d}: Train loss={tr_loss:.4f} acc={tr_acc:.4f} AUC={tr_auc:.4f} | '
                  f'Val loss={va_loss:.4f} acc={va_acc:.4f} AUC={va_auc:.4f}')
    return history


def save_plots(history, name):
    tr_l, tr_a, tr_u, va_l, va_a, va_u = zip(*history)
    xs = np.arange(len(tr_l))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, y1, y2, ylabel in zip(axes,
                                   [tr_l, tr_a, tr_u],
                                   [va_l, va_a, va_u],
                                   ['loss', 'accuracy', 'AUC']):
        ax.plot(xs, y1, label='train'); ax.plot(xs, y2, label='val')
        ax.set_ylabel(ylabel); ax.legend()
    fig.suptitle(f'Graph Classification ({name} pooling)')
    path = f'plots/graph_class_{name}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def run_pooling(pooling_name):
    print(f'\n=== Pooling: {pooling_name} ===')
    model     = GraphClassifier(HIDDEN_DIM, OUTPUT_DIM, NUM_GCN_LAYERS,
                                DROPOUT_PCT, pooling=pooling_name).to(DEVICE)
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history   = train_loop(model, optimizer, loss_fn)
    save_plots(history, pooling_name)
    _, test_acc, test_lp = eval_step(model, loss_fn, test_loader)
    test_auc = compute_roc_auc(test_lp)
    print(f'  Test accuracy: {test_acc:.5f} | Test AUC: {test_auc:.5f}')
    return test_acc, test_auc


if __name__ == '__main__':
    results = {}
    for pooling in ['mean', 'max', 'sum']:
        acc, auc = run_pooling(pooling)
        results[pooling] = (acc, auc)

    print('\n=== Summary ===')
    for p, (acc, auc) in results.items():
        print(f'  {p:4s} pooling | Test acc={acc:.5f} AUC={auc:.5f}')
    print('Done. Plots saved to plots/graph_class_*.png')
