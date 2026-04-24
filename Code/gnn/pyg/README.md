# PyTorch Geometric (PyG) - GNN Teaching Examples

Educational Python scripts demonstrating Graph Neural Networks using [PyTorch Geometric](https://pyg.org/).
Each script is self-contained and runnable end-to-end.

## Environment Setup

Uses the `genai` conda environment with **CUDA 11.8** and **PyTorch 2.7**:

```bash
conda activate genai
```

To install dependencies from scratch:
```bash
# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.7.0+cu118.html

# Other dependencies
pip install captum trimesh scikit-learn matplotlib networkx seaborn tqdm
```

## Scripts

All scripts auto-download their datasets to `data/` on first run (except FAUST, see below).
GPU is used automatically when available.

### PyTorch Fundamentals (no graph structure)

| Script | Description | Dataset |
|--------|-------------|---------|
| `pytorch_linear_regression.py` | Linear regression: learn f = 2*x via SGD | Synthetic |
| `pytorch_mnist_mlp.py` | MLP for digit classification | MNIST (auto-download) |
| `pytorch_cifar_cnn.py` | CNN for image classification | CIFAR-10 (auto-download) |

### Graph Neural Networks

| Script | Description | Dataset |
|--------|-------------|---------|
| `gnn_intro_karate_gcn.py` | GCN intro: node community detection | KarateClub (built-in) |
| `gnn_node_classification_cora.py` | MLP vs GCN vs GAT on citation network | Cora (auto-download) |
| `gnn_graph_classification_mutag.py` | Graph-level mutagenicity prediction | MUTAG (auto-download) |
| `gnn_scaling_pubmed.py` | Mini-batch training with ClusterData | PubMed (auto-download) |
| `gnn_pointcloud_classification.py` | Custom PointNet message-passing on 3D shapes | GeometricShapes (auto-download) |
| `gnn_explanation_captum.py` | Saliency + Integrated Gradients attribution | Mutagenicity (auto-download) |
| `gnn_mesh_segmentation_faust.py` | Vertex-level body-part segmentation | FAUST (manual, see below) |

## Running Scripts

```bash
conda activate genai
cd Code/gnn/pyg

python pytorch_linear_regression.py
python pytorch_mnist_mlp.py
python pytorch_cifar_cnn.py

python gnn_intro_karate_gcn.py
python gnn_node_classification_cora.py
python gnn_graph_classification_mutag.py
python gnn_scaling_pubmed.py
python gnn_pointcloud_classification.py
python gnn_explanation_captum.py
python gnn_mesh_segmentation_faust.py   # requires FAUST dataset
```

Visualization outputs are saved to `plots/`.

## FAUST Dataset (gnn_mesh_segmentation_faust.py)

The MPI-FAUST dataset requires free registration:

1. Register at http://faust.is.tue.mpg.de/
2. Download the dataset zip file
3. Place it at `data/MPI-FAUST.zip`
4. Run `python gnn_mesh_segmentation_faust.py` - it auto-extracts the zip

## Key Concepts Covered

- **GCNConv** - Graph Convolutional Network (Kipf & Welling, 2017)
- **GATConv** - Graph Attention Network (Velickovic et al., 2018)
- **GraphConv** - General graph convolution with edge weight support
- **MessagePassing** - Custom message-passing for PointNet and FeastNet
- **ClusterData/ClusterLoader** - Mini-batch training for large graphs
- **global_mean/max_pool** - Graph-level readout for graph classification
- **Captum Saliency/IntegratedGradients** - GNN prediction attribution
- **FeatureSteeredConvolution** - FeastNet 3D mesh convolution (Verma et al., 2018)

## Data Directory

Datasets are downloaded/stored in `data/`:
```
data/
  Planetoid/       # Cora, PubMed
  TUDataset/       # MUTAG, Mutagenicity
  GeometricShapes/ # 40 geometric shape classes
  MPI-FAUST/       # Human body meshes (manual download)
  MNIST/           # MNIST digits
  cifar-10-batches-py/  # CIFAR-10
```
