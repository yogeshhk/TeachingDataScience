# Graph Neural Networks (GNN)

Python scripts and notebooks covering Graph Neural Networks using PyTorch Geometric (PyG), with applications to molecular property prediction and knowledge graphs.

## Sub-projects

| Directory | Description |
|-----------|-------------|
| `pyg/` | PyTorch Geometric tutorials: GCN, GAT, GraphSAGE, message passing |
| `odsc2021-sujitpal/` | ODSC 2021 workshop by Sujit Pal: link prediction and node classification |
| `gnn-project-deepfindr/` | End-to-end GNN project from the Deepfindr series |
| `molecule-deepfindr/` | Molecular property prediction on graph-structured chemical data |

## Setup

`pyg/` has no `environment.yml`; its `README.md` describes the `genai` conda environment and the
PyG install steps (PyG needs torch and CUDA versions that match; see pyg.org/install).
`molecule-deepfindr/` has its own `environment.yml`:

```bash
conda env create -f molecule-deepfindr/environment.yml
conda activate genai
```

## Key Concepts

- Graph representation: nodes, edges, node features, edge attributes
- Message passing: aggregate neighbour features → update node embedding
- GCN / GAT / GraphSAGE for node classification and link prediction
- Global pooling for graph-level property prediction (molecular graphs)
- Captum for GNN explainability

## References

- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
- [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
