# Graphs

* Data Structure consisting of nodes (also known as Vertices) and edges, _G = (V, E)_
* Can represent real-world objects, examples:
  * words in a document,  
  * documents in a citation network,
  * people and organizations in a social media network, 
  * atoms in molecular structure
* Matrix Representation: Adjacency Matrix _A_
  * For graph with _n_ nodes, _A_ has shape _(n, n)_
  * If nodes _i_ and _j_ are connected, _A(i, j)_ and _A(j, i)_ = 1
  * Minor variations for directed graphs (_A(i, j) != A(j, i)_) and weighted graphs (_A(i, j) = w_)
* Graphs can optionally have node features _X_
  * For graph with _n_ nodes and feature vector of size _f_, _X_ has shape _(n, f)_
* Graphs can optionally also have edge features 

---

# Machine Learning Models

* Goal: Learn a mapping _F_ from an input space _X_ to an output space _y_
* Hypothesize some model M with random weights _θ_
* Formulate the task as an optimization problem
  
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;\min_{\theta}&space;\mathcal{L}(y,&space;\mathcal{F}(x))" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_jvn&space;\min_{\theta}&space;\mathcal{L}(y,&space;\mathcal{F}(x))" title="\min_{\theta} \mathcal{L}(y, \mathcal{F}(x))" /></a>
</p>

* Use gradient descent to update the model weights until convergence

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;\theta&space;\leftarrow&space;\theta&space;-&space;\eta&space;\nabla_{\theta}&space;\mathcal{L}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_jvn&space;\theta&space;\leftarrow&space;\theta&space;-&space;\eta&space;\nabla_{\theta}&space;\mathcal{L}" title="\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}" /></a>
</p>

* Test fitted model for accuracy on new data, try a different model M if needed
---

# Graph Models for Machine Learning

* ML and DL tools are optimized for simple structures
  * Convolutional Neural Networks
    * images
    * regular lattice structures
  * Recurrent Neural Networks
    * text and sequence data
    * time series data
* Problems with graphs
  * Topological complexity
  * Indeterminate size
  * Not permutation invariant
  * Instances not independent of each other

---

# Extending Convolutions to Graphs

* [Convolutions in CNN](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
* Extending to graph - aggregate information from neighboring nodes

<p align="center">
<img src="figures/fig-01-01.png"/><br/>
<b>Source: CS-224W slide 06-GNN-1.pdf</b>
</p>

---

# Deep Learning architecture for images

* Multiple layers of convolution + non-linearity + pooling
* Fully connected layer(s) with non-linearity converts feature map to output prediction 
* Loss function Cross Entropy for classification, Mean Squared Error for regression
* Uses gradient descent to optimize loss function

<p align="center">
<img src="figures/fig-01-02.jpg"/>
</p>

---

# Deep Learning architecture for graphs

* Each graph convolution layer corresponds to aggregating 1-hop neighborhood info for each node
* In a GNN with k convolution layers, each node has information about nodes k-hops away
* Value of k dictated by application, usually small (unlike giant CNNs)

<p align="center">
<img src="figures/fig-01-03.png"/><br/>
<b>Source: CS-224W slide 06-GNN-1.pdf</b>
</p>

---

# Computation Graph

* Aggregate information from neighbors
* Apply neural network on aggregated information (gray boxes)
* Each node defines computation graph based on its neighborhood, i.e. each node has its own neural network!
* Achieved using __message passing__

<p align="center">
<img src="figures/fig-01-04.png"/><br/>
<b>Source: CS-224W slide 06-GNN-1.pdf</b>
</p>

---

# Message Passing

* Elegant approach to handle irregularity (diversity of computation graphs) in GNN
* Message Passing steps
  * For each node in graph, _gather_ all neighbor node embeddings (messages)
  * Aggregate all messages via an aggregate function (such as sum or average)
  * All pooled messages are passed through _update_ function, usually a learned neural network
* Reference: section __Passing messages between parts of the graph__ in [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
* More coverage of Message Passing in Part III of tutorial

---

# Pytorch Geometric (PyG)

* From the [Pytorch documentation](https://pytorch-geometric.readthedocs.io/en/latest/index.html):

> PyG (PyTorch Geometric) is a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.
> 
> It consists of various methods for deep learning on graphs and other irregular structures, also known as geometric deep learning, from a variety of published papers. In addition, it consists of easy-to-use mini-batch loaders for operating on many small and single giant graphs, multi GPU-support, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

* Provides following graph abstractions to Pytorch
  * Graph Layers ([torch_geometric.nn](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html))
  * Graph Dataloaders ([torch_geometric.loader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html))
  * Graph Transforms ([torch_geometric.transforms](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html))
  * Popular Graph Datasets ([torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html))

<p align="center">
<img src="figures/fig-01-05.png" width="200" height="200"/>
<img src="figures/fig-01-06.png" width="150" height="150"/>
</p>

---

# Popular PyG Graph Layers

* Main difference is aggregation strategy

* __Graph Convolution Network (GCN)__
  * Aggregate self-features and neighbor features

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;h_v^{(k&plus;1)}&space;=&space;\sigma(W_k&space;\sum_{u&space;\in&space;N(v)}&space;\frac{h_u^{(k)}}{|N(v)|})&space;&plus;&space;B_k&space;h_v^{(k)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_jvn&space;h_v^{(k&plus;1)}&space;=&space;\sigma(W_k&space;\sum_{u&space;\in&space;N(v)}&space;\frac{h_u^{(k)}}{|N(v)|})&space;&plus;&space;B_k&space;h_v^{(k)}" title="h_v^{(k+1)} = \sigma(W_k \sum_{u \in N(v)} \frac{h_u^{(k)}}{|N(v)|}) + B_k h_v^{(k)}" /></a>
</p>

* __Graph Attention Network (GAT)__
  * Aggregate neighboring features with weights derived from attention mechanism (between self-features and all neighbor features)
  * Attention computed using [Bahdanau model](https://arxiv.org/abs/1409.0473) using feedforward network

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;\begin{align*}&space;a_{ij}&space;&=&space;attention(h_i,&space;h_j)&space;\\&space;\alpha_{ij}&space;&=&space;\frac{exp(a_{ij})}{\sum_{k&space;\in&space;N(i)}&space;exp(a_{ik})}&space;\\&space;h_v^{(k&plus;1)}&space;&=&space;\sigma(\sum_{i&space;\in&space;N(v)}&space;\alpha_{iv}&space;W&space;h_i^{(k)})&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_jvn&space;\begin{align*}&space;a_{ij}&space;&=&space;attention(h_i,&space;h_j)&space;\\&space;\alpha_{ij}&space;&=&space;\frac{exp(a_{ij})}{\sum_{k&space;\in&space;N(i)}&space;exp(a_{ik})}&space;\\&space;h_v^{(k&plus;1)}&space;&=&space;\sigma(\sum_{i&space;\in&space;N(v)}&space;\alpha_{iv}&space;W&space;h_i^{(k)})&space;\end{align*}" title="\begin{align*} a_{ij} &= attention(h_i, h_j) \\ \alpha_{ij} &= \frac{exp(a_{ij})}{\sum_{k \in N(i)} exp(a_{ik})} \\ h_v^{(k+1)} &= \sigma(\sum_{i \in N(v)} \alpha_{iv} W h_i^{(k)}) \end{align*}" /></a>
</p>

* __GraphSAGE (SAmple and AggreGatE)__
  * Sample a subset of neighbors instead of using all of them (for scalability)
  * Importance sampling
    * Define node neighborhood using random walks
    * Sum up importance scores generated by random walks
  * Can use MEAN, MAX or SUM as aggregate functions
* __Graph Isomorphisnm Network (GIN)__
  * uses SUM aggregation because it is better than MEAN and MAX aggregation for detecting graph similarity (isomorphism)
* Available as [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv), [SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv), [GATConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv) and [GINConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv)

---

# PyG DataLoaders

* Extension of Pytorch DataLoaders
* DataLoader ([torch_geoemtric.loader.DataLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.DataLoader)) -- merges Data objects from a Dataset to a mini-batch
* Dataset ([torch_geometric.data.Dataset](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset) -- wrapper creating graph datasets
* Data ([torch_geometric.data.Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) -- represents a single graph in PyG, has following attributes by default.
  * `data.x` -- node feature matrix with shape `(num_nodes, num_node_features)`
  * `data.edge_index` -- edges in COO (coordinate) format with shape `(2, num_edges)`
  * `data.edge_attr` -- edge feature matrix with shape `(num_edges, num_edge_features)`
  * `data.y` -- target matrix with shape `(num_nodes, *)`
  * `data.pos` -- node position matrix with shape `(num_nodes, num_dimensions)`
* Parallelization over mini-batch achieved by creating block diagonal adjacency matrices (defined by `edge_index`), concatenating feature and target matrices in the node dimension, allows handling different number of nodes and edges over examples in single batch

<p align="center">
<img src="figures/fig-01-07.png"/>
</p>

---

# GNN Applications

* __Node classification__
  * Supervised -- labeling items (represented as nodes in a graph) by looking at the labels of their samples.
  * Unsupervised -- use random walk based embeddings or other graph features to generate labels
* __Graph classification__
  * Classify a graph into one of several categories
  * Examples -- determine if a protein is an enzyme, chemical is toxic, categorizing documents (NLP)
* __Link Prediction__
  * Predicts if there is a connection between two entities in a graph
  * Example -- Infer / Predict social connections in social network graph
* __Graph clustering__
  * Use GNN (without classifier head) as encoder then cluster feature maps
* __Generative Graph Models__
  * Use Variational AutoEncoder (VAE) that learns to predict graph's Adjacency Matrix (like images)
  * Build graph sequentially, starting with subgraph and applying nodes and edges sequentially

