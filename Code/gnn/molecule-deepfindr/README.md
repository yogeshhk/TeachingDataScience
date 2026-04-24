# Hands on with Graph Neural Networks

Graph-level regression with PyTorch Geometric on the ESOL molecular solubility dataset.  
Original notebook: <https://colab.research.google.com/drive/16GBgwYR2ECiXVxA1BoLxYshKczNMeEAQ>

## What it does

| Step | Detail |
|------|--------|
| Dataset | **ESOL** (1128 organic molecules, DeepChem/MoleculeNet) — labels are log water-solubility |
| Input | SMILES string -> molecular graph (nodes = atoms, edges = bonds, 9 node features) |
| Model | 4-layer **GCN** + concatenated global-max/mean pooling + Linear head |
| Task | Graph-level regression (predict log mol/L) |
| Outputs | `molecule_sample.png`, `training_loss.png`, `predictions.png` |

## Environment

Tested on:
- Windows 11, Python 3.10
- NVIDIA GeForce MX570 A (2 GB VRAM), driver 595.71, CUDA 13.2
- torch 2.7.1+cu118, torch-geometric 2.6.1, rdkit 2024.9.6

Uses the existing **`genai`** conda environment.

## Setup

### 1. Activate the genai environment

```bat
conda activate genai
```

### 2. Install PyTorch (CUDA 11.8 build — works with any driver >= 11.8)

Skip if torch is already installed:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install PyTorch Geometric and its C++ extensions

```bat
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv ^
    -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
```

### 4. Install remaining dependencies

```bat
pip install rdkit seaborn pandas "matplotlib>=3.9"
```

> **Note:** matplotlib >= 3.9 is required for numpy 2.x compatibility.

### All-in-one (if starting fresh)

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv ^
    -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
pip install rdkit seaborn pandas "matplotlib>=3.9"
```

## Run

```bat
conda activate genai
cd Code/gnn
python deepfindr_graphneuralnets.py
```

Training runs for **2000 epochs** (~5-10 min on MX570 A). Progress is printed every 100 epochs.

## Output files

| File | Description |
|------|-------------|
| `molecule_sample.png` | RDKit 2D structure of first molecule |
| `training_loss.png` | MSE loss curve over 2000 epochs |
| `predictions.png` | Scatter: predicted vs actual solubility on test set |
| `ESOL/` | Cached dataset (auto-downloaded on first run) |

## Key fixes vs original Colab notebook

| Issue | Fix |
|-------|-----|
| `!pip install` / `!nvidia-smi` shell magic | Removed — not valid in `.py` scripts |
| Colab rdkit-via-miniconda installer | Removed — `rdkit` is now on PyPI |
| `torch==1.6.0` hard-pin | Removed — uses whatever torch is in the env |
| `from torch_geometric.data import DataLoader` | Updated to `torch_geometric.loader` |
| `data.len` | Fixed to `len(data)` |
| `F.tanh(x)` | Changed to `torch.tanh(x)` (preferred) |
| `sns.lineplot(x_data, y_data)` positional args | Changed to `sns.lineplot(x=..., y=...)` |
| `IPythonConsole` import | Removed (Jupyter-only) |
| `molecule` bare expression for display | Changed to `Draw.MolToImage(...).save(...)` |
| No `plt.show()` / `plt.savefig()` | Added save + show for all plots |
| `batch.y` shape mismatch | Added `batch.y[:, 0:1]` guard for multi-task y |
| matplotlib 3.8.x vs numpy 2.x | Upgraded to matplotlib 3.10+ |

## Model architecture

```
GCN(
  initial_conv : GCNConv(9 -> 64)
  conv1        : GCNConv(64 -> 64)
  conv2        : GCNConv(64 -> 64)
  conv3        : GCNConv(64 -> 64)
  out          : Linear(128 -> 1)    # 128 = concat(global_max, global_mean)
)
Total parameters: 33,217
```

## Further improvements

- Add dropout layers between GCN layers
- Try `TopKPooling` or `SAGPooling` instead of global pooling
- Add batch normalisation (not on the final head when batch_size=1)
- Hyperparameter search for `lr`, `embedding_size`, `num_layers`
- Report RMSE on a held-out test set per epoch
- See more examples: <https://github.com/rusty1s/pytorch_geometric/tree/master/examples>
