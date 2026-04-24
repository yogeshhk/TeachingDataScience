# -*- coding: utf-8 -*-
"""
Hands on with Graph Neural Networks
Based on: https://colab.research.google.com/drive/16GBgwYR2ECiXVxA1BoLxYshKczNMeEAQ

Dataset : ESOL - water solubility for 1128 compounds (DeepChem / MoleculeNet)
Task    : Graph-level regression - predict log-solubility from molecular graph
Model   : 4-layer GCN with global max+mean pooling -> Linear head
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.nn import Linear

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw

from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"Device  : {device}")
print()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# ESOL: 1128 small organic molecules labelled with water solubility (log mol/L)
# Each molecule is encoded as a graph: nodes = atoms, edges = bonds
data = MoleculeNet(root=".", name="ESOL")

print("Dataset type     :", type(data))
print("Node features    :", data.num_features)
print("Dataset length   :", len(data))
print("Sample           :", data[0])
print("Sample nodes     :", data[0].num_nodes)
print("Sample edges     :", data[0].num_edges)
print("Node feat shape  :", data[0].x.shape)   # [num_nodes, 9]
print("Edge index shape :", data[0].edge_index.t().shape)
print("Target shape     :", data[0].y.shape)
print()

# ---------------------------------------------------------------------------
# Molecule visualisation (script-safe: saves PNG instead of Jupyter display)
# ---------------------------------------------------------------------------
smiles = data[0]["smiles"]
print("Sample SMILES:", smiles)
molecule = Chem.MolFromSmiles(smiles)
mol_img = Draw.MolToImage(molecule, size=(300, 200))
mol_img.save("molecule_sample.png")
print("Molecule image saved -> molecule_sample.png")
print()

# ---------------------------------------------------------------------------
# GCN model
# ---------------------------------------------------------------------------
EMBEDDING_SIZE = 64

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)

        self.initial_conv = GCNConv(data.num_features, EMBEDDING_SIZE)
        self.conv1 = GCNConv(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.conv2 = GCNConv(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.conv3 = GCNConv(EMBEDDING_SIZE, EMBEDDING_SIZE)
        # Concat global-max and global-mean -> double width
        self.out = Linear(EMBEDDING_SIZE * 2, 1)

    def forward(self, x, edge_index, batch_index):
        hidden = torch.tanh(self.initial_conv(x, edge_index))
        hidden = torch.tanh(self.conv1(hidden, edge_index))
        hidden = torch.tanh(self.conv2(hidden, edge_index))
        hidden = torch.tanh(self.conv3(hidden, edge_index))
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)
        return self.out(hidden), hidden


model = GCN().to(device)
print(model)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(data[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):],
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def train():
    model.train()
    last_loss = None
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
        # Newer MoleculeNet stores y as [N, num_tasks]; take column 0 for ESOL
        y = batch.y[:, 0:1] if batch.y.dim() > 1 else batch.y.unsqueeze(1)
        loss = loss_fn(pred, y.float())
        loss.backward()
        optimizer.step()
        last_loss = loss
    return last_loss


print("Starting training (2000 epochs)...")
losses = []
for epoch in range(2000):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Train Loss {loss.item():.4f}")

print()

# ---------------------------------------------------------------------------
# Training loss plot
# ---------------------------------------------------------------------------
losses_float = [float(l.cpu().detach()) for l in losses]
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=range(len(losses_float)), y=losses_float, ax=ax)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("GCN Training Loss on ESOL")
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()
print("Training loss plot saved -> training_loss.png")
print()

# ---------------------------------------------------------------------------
# Test predictions
# ---------------------------------------------------------------------------
model.eval()
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch = test_batch.to(device)
    pred, _ = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    y_real = test_batch.y[:, 0] if test_batch.y.dim() > 1 else test_batch.y
    df = pd.DataFrame({
        "y_real": y_real.cpu().tolist(),
        "y_pred": pred.cpu().squeeze().tolist(),
    })

print("Test batch predictions (first 10 rows):")
print(df.head(10).to_string(index=False))
print()

fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=df, x="y_real", y="y_pred", ax=ax)
lims = (-7, 2)
ax.plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual log-solubility")
ax.set_ylabel("Predicted log-solubility")
ax.set_title("Predicted vs Actual (Test Set)")
ax.legend()
plt.tight_layout()
plt.savefig("predictions.png", dpi=150)
plt.show()
print("Predictions scatter plot saved -> predictions.png")
