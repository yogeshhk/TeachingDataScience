# Heterogeneous Graphs

* So far we have assumed that graphs are __homogenous__, i.e., have only one kind of node and relation, i.e. _G = (V, E)_
* Graphs in real world are often __heterogeneous__, i.e, they can have multiple entity types and relation types, i.e. _G = (V, E, R, T)_
* Examples of heterogeneous grophs -- Knowledge Graphs, Social Graphs, etc.
* Relational GCN (`torch_geometric.nn.RGCN`) -- network learns individual sets of weight matrices for each relation type in graph instead of one set of weights.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;h_v^{(l&plus;1)}&space;=&space;\sigma&space;\left&space;(&space;\sum_{r&space;\in&space;R}&space;\sum_{u&space;\in&space;N_p^r]}&space;\frac{1}{\left&space;\|&space;N_p^r&space;\right&space;\|}&space;W_r^{(l)}h_u^{(l)}&space;&plus;&space;W_0^{(l)}h_v^{(l)}\right&space;)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_jvn&space;h_v^{(l&plus;1)}&space;=&space;\sigma&space;\left&space;(&space;\sum_{r&space;\in&space;R}&space;\sum_{u&space;\in&space;N_p^r]}&space;\frac{1}{\left&space;\|&space;N_p^r&space;\right&space;\|}&space;W_r^{(l)}h_u^{(l)}&space;&plus;&space;W_0^{(l)}h_v^{(l)}\right&space;)" title="h_v^{(l+1)} = \sigma \left ( \sum_{r \in R} \sum_{u \in N_p^r]} \frac{1}{\left \| N_p^r \right \|} W_r^{(l)}h_u^{(l)} + W_0^{(l)}h_v^{(l)}\right )" /></a>
</p>

* Heterogeneous GCN (`torch_geometric.nn.HeteroConv`) -- network learns different weights for each unique (h, r, t) triple by type, using individual GNN layers.

```python
hetero_conv = HeteroConv({
    ('paper', 'cites', 'paper'): GCNConv(-1, 64),
    ('author', 'writes', 'paper'): SAGEConv(-1, 64),
    ('paper', 'written_by', 'author'): GATConv(-1, 64),
}, aggr='sum')
```

---

# Custom Layers

* Pytorch Geometric provides a comprehensive list of different kinds of Graph layers in `torch_geometric.nn`
* Operations in a graph layer can be thought of as a scatter-gather (or message-aggregation) operation.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;\begin{align*}&space;m_u^{(l)}&space;&=&space;MSG^{(l)}&space;\left&space;(&space;h_u^{(l-1)}&space;\right&space;),&space;u&space;\in&space;\left&space;\{&space;N(v)&space;\cup&space;v&space;\right&space;\}&space;\\&space;h_v^{(l)}&space;&=&space;AGG^{(l)}&space;\left&space;(&space;\left&space;\{&space;m_u^{(l)},&space;u&space;\in&space;N(v)&space;\right&space;\},&space;m_v^{(l)}&space;\right&space;)&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_jvn&space;\begin{align*}&space;m_u^{(l)}&space;&=&space;MSG^{(l)}&space;\left&space;(&space;h_u^{(l-1)}&space;\right&space;),&space;u&space;\in&space;\left&space;\{&space;N(v)&space;\cup&space;v&space;\right&space;\}&space;\\&space;h_v^{(l)}&space;&=&space;AGG^{(l)}&space;\left&space;(&space;\left&space;\{&space;m_u^{(l)},&space;u&space;\in&space;N(v)&space;\right&space;\},&space;m_v^{(l)}&space;\right&space;)&space;\end{align*}" title="\begin{align*} m_u^{(l)} &= MSG^{(l)} \left ( h_u^{(l-1)} \right ), u \in \left \{ N(v) \cup v \right \} \\ h_v^{(l)} &= AGG^{(l)} \left ( \left \{ m_u^{(l)}, u \in N(v) \right \}, m_v^{(l)} \right ) \end{align*}" /></a>
</p>

* Custom graph layers possible by extending `torch_geometric.nn.MessagePassing` (see [Creating Message Passing Networks](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) for example code)
* [Exercise: Create custom Graph Attention (GAT) layer.](06x-custom-layer.ipynb)

---

# Custom Datasets

* Pytorch Geometric and other platforms offer large array of useful graph datasets 
  * [PyG datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
  * [OGB datasets](https://ogb.stanford.edu/docs/dataset_overview/)
* Dataset and DataLoader abstraction useful for working with PyG networks easy.
* Creating DataLoader
  * From Dataset -- `DataLoader(dataset, ...)`
  * From Data -- `DataLoader([Data, ...], ...)`
* Creating custom Dataset
  * In Memory Dataset -- extend `torch_geometric.data.InMemoryDataset`
  * On Disk Dataset -- extend `torch_geometric.data.Dataset`
* [Example code to create custom dataset](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html)
* [Video: Creating a Custom Dataset in Pytorch Geometric](https://www.youtube.com/watch?v=QLIkOtKS4os) by DeepFindr
* [Exercise: Create custom dataset](07x-custom-dataset.ipynb)

---
