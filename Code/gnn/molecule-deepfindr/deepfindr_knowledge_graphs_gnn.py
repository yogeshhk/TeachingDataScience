# -*- coding: utf-8 -*-
"""
Node Classification on large Knowledge Graphs (Cora dataset)
Based on: https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX

Dataset:
- Cora: 2708 scientific publications, 7 classes
- Nodes = papers, Edges = citations
- Node features = 0/1 word-presence vectors (1433 unique words)
- Labels: Neural_Networks, Rule_Learning, Reinforcement_Learning,
          Probabilistic_Methods, Theory, Genetic_Algorithms, Case_Based
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.nn import Linear
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print(f'Number of graphs    : {len(dataset)}')
print(f'Number of features  : {dataset.num_features}')
print(f'Number of classes   : {dataset.num_classes}')
print(50 * '=')

# Single large graph — use binary train/val/test masks for node-level tasks
data = dataset[0]
print(data)
print(f'Number of nodes           : {data.num_nodes}')
print(f'Number of edges           : {data.num_edges}')
print(f'Number of training nodes  : {data.train_mask.sum()}')
print(f'Training node label rate  : {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Is undirected             : {data.is_undirected()}')
print()

# Only ~140 labelled training nodes (20 per class) — graph structure is crucial
print('Node feature shape  :', data.x.shape)   # [2708, 1433]
print('First node features (first 50):', data.x[0][:50])
print('Node labels:', data.y)
print('Test mask len == num_nodes:', len(data.test_mask) == data.num_nodes)
print('Edge index (first 10):\n', data.edge_index.t()[:10])
print()

# ---------------------------------------------------------------------------
# GCN model for node classification
# ---------------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return F.softmax(self.out(x), dim=1)


model = GCN(hidden_channels=16)
print(model)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    return int(test_correct.sum()) / int(data.test_mask.sum())


losses = []
for epoch in range(0, 1001):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# ---------------------------------------------------------------------------
# Training loss plot
# ---------------------------------------------------------------------------
losses_float = [float(l.cpu().detach().numpy()) for l in losses]
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(len(losses_float))), y=losses_float, ax=ax)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("GCN Training Loss on Cora")
plt.tight_layout()
plt.savefig("training_loss_kg.png", dpi=150)
plt.show()
print("Training loss saved -> training_loss_kg.png")
print()

# ---------------------------------------------------------------------------
# Test accuracy
# ---------------------------------------------------------------------------
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
print()

# ---------------------------------------------------------------------------
# Prediction bar for a sample node
# ---------------------------------------------------------------------------
sample = 9
sns.set_theme(style="whitegrid")
with torch.no_grad():
    pred_all = model(data.x, data.edge_index)
print('Prediction shape:', pred_all.shape)

fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(x=np.array(range(dataset.num_classes)),
            y=pred_all[sample].detach().cpu().numpy(), ax=ax)
ax.set_xlabel("Class")
ax.set_ylabel("Probability")
ax.set_title(f"Node {sample} — class probabilities")
plt.tight_layout()
plt.savefig("node_prediction_kg.png", dpi=150)
plt.show()
print("Node prediction saved -> node_prediction_kg.png")
print()

# ---------------------------------------------------------------------------
# TSNE embedding visualisation
# ---------------------------------------------------------------------------
from sklearn.manifold import TSNE


def visualize(h, color, epoch):
    fig = plt.figure(figsize=(5, 5), frameon=False)
    fig.suptitle(f'Epoch = {epoch}')
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70,
                c=color.detach().cpu().numpy(), cmap="Set2")
    fig.canvas.draw()
    # buffer_rgba() replaces the deprecated tostring_rgb() in matplotlib 3.6+
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h_px = fig.canvas.get_width_height()
    rgba = buf.reshape((h_px, w, 4))
    plt.close(fig)
    return rgba[:, :, :3]  # drop alpha channel


# Reset model weights for a fresh visualisation run
for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

print("Training with TSNE snapshots (2000 epochs)...")
images = []
for epoch in range(0, 2000):
    loss = train()
    if epoch % 50 == 0:
        out = model(data.x, data.edge_index)
        images.append(visualize(out, color=data.y, epoch=epoch))
print("TSNE visualisation finished.")
print()

# ---------------------------------------------------------------------------
# Save embeddings as GIF (requires moviepy)
# ---------------------------------------------------------------------------
try:
    from moviepy.editor import ImageSequenceClip
    filename = "embeddings.gif"
    clip = ImageSequenceClip(images, fps=1)
    clip.write_gif(filename, fps=1)
    print(f"Embedding GIF saved -> {filename}")
except ImportError:
    print("moviepy not installed — skipping GIF. Install with: pip install moviepy")
