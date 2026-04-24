"""
GNN Link Prediction: Cora Citation Network
Predicts missing links using GCN node embeddings with dot-product similarity.
Uses RandomLinkSplit to create positive/negative edge train/val/test splits.
Source: ODSC West 2021 tutorial by Sujit Pal (solution notebook 04)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score

os.makedirs('data/cora', exist_ok=True)
os.makedirs('plots', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Hyperparameters
HIDDEN_DIM     = 128
NUM_GCN_LAYERS = 3
DROPOUT_PCT    = 0.5
LEARNING_RATE  = 1e-2
NUM_EPOCHS     = 100
LOG_EVERY      = 10


# ── Data ─────────────────────────────────────────────────────────────────────

transform = T.Compose([
    T.NormalizeFeatures(),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                      is_undirected=True, add_negative_train_samples=True),
])
dataset = Planetoid(root='data/cora', name='Cora', transform=transform)

train_data, val_data, test_data = dataset[0]
print(f'Dataset: Cora (link prediction split)')
print(f'  Features: {dataset.num_features}')
print(f'  Train edges: {train_data.edge_label_index.shape[1]}')
print(f'  Val edges:   {val_data.edge_label_index.shape[1]}')
print(f'  Test edges:  {test_data.edge_label_index.shape[1]}')

INPUT_DIM  = dataset.num_features
OUTPUT_DIM = dataset.num_classes


# ── Model ────────────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """
    GCN encoder + dot-product edge scorer.
    Node embeddings are learned; link probability = sigmoid(u · v).
    """

    def __init__(self, input_dim, hidden_dim, num_graph_layers, dropout_pct):
        super().__init__()
        self.dropout_pct = dropout_pct
        self.num_graph_layers = num_graph_layers

        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        for _ in range(num_graph_layers - 1):
            self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_graph_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_label_index = (
            data.x, data.edge_index, data.edge_label_index)
        for i in range(self.num_graph_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_pct, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        # Dot-product edge scores
        src = x[edge_label_index[0]]
        dst = x[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def loss(self, pred, label):
        return F.binary_cross_entropy_with_logits(pred, label)


# ── Training helpers ─────────────────────────────────────────────────────────

def train_step(model, optimizer, data):
    model.train()
    data = data.to(DEVICE)
    optimizer.zero_grad()
    pred  = model(data)
    label = data.edge_label
    loss  = model.loss(pred, label)
    loss.backward()
    optimizer.step()
    auc = roc_auc_score(label.detach().cpu().numpy(),
                        pred.detach().cpu().numpy())
    return loss.item(), auc


def eval_step(model, data):
    model.eval()
    data = data.to(DEVICE)
    with torch.no_grad():
        pred  = model(data)
        label = data.edge_label
        loss  = model.loss(pred, label)
        auc   = roc_auc_score(label.cpu().numpy(), pred.cpu().numpy())
    return loss.item(), auc


def train_loop(model, optimizer, num_epochs=NUM_EPOCHS):
    history = []
    for epoch in range(num_epochs):
        tr_loss, tr_auc = train_step(model, optimizer, train_data)
        va_loss, va_auc = eval_step(model, val_data)
        history.append((tr_loss, tr_auc, va_loss, va_auc))
        if epoch == 0 or (epoch + 1) % LOG_EVERY == 0:
            print(f'  Epoch {epoch+1:3d}: Train loss={tr_loss:.4f} AUC={tr_auc:.4f} | '
                  f'Val loss={va_loss:.4f} AUC={va_auc:.4f}')
    return history


def save_plots(history, filename='plots/link_pred_history.png'):
    tr_l, tr_u, va_l, va_u = zip(*history)
    xs = np.arange(len(tr_l))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(xs, tr_l, label='train'); ax1.plot(xs, va_l, label='val')
    ax1.set_ylabel('loss'); ax1.legend()
    ax2.plot(xs, tr_u, label='train'); ax2.plot(xs, va_u, label='val')
    ax2.set_ylabel('AUC'); ax2.legend()
    fig.suptitle('Link Prediction on Cora')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


if __name__ == '__main__':
    model = LinkPredictor(INPUT_DIM, HIDDEN_DIM, NUM_GCN_LAYERS,
                          DROPOUT_PCT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('\nTraining link predictor...')
    history = train_loop(model, optimizer)
    save_plots(history)

    _, test_auc = eval_step(model, test_data)
    print(f'\nTest AUC: {test_auc:.5f}')
    print('Done.')
