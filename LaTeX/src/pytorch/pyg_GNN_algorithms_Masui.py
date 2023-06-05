# https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275
import random

import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

dataset = Planetoid(root='/tmp/Cora', name='Cora')
graph = dataset[0]


def convert_to_networkx(graph, n_sample=None):
    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    return g, y


def plot_graph(g, y):
    plt.figure(figsize=(9, 7))
    nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
    plt.show()


# g, y = convert_to_networkx(graph, n_sample=1000)
# plot_graph(g, y)

# Nodes: 2708, each with features length 1433
# Edges, node index pairs
# Each node has one of seven classes which is going to be our model target/label. ie y

# splitting the nodes into train, valid, and test

split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
graph = split(graph)

# Please note the data splits are written into mask attributes in the graph object (see the image below) instead of
# splitting the graph itself. Those masks are used for training loss calculation and model evaluation, whereas graph
# convolutions use entire graph data.

# Baseline performance, without considering graph or connectivity, but just Node features

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dataset.num_node_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, dataset.num_classes)
        )

    def forward(self, data):
        x = data.x  # only using node features (x)
        output = self.layers(x)
        return output


def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, graph, graph.val_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    return model


def eval_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc


import torch

device = "cpu"
# mlp = MLP().to(device)
# optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
# criterion = nn.CrossEntropyLoss()
# mlp = train_node_classifier(mlp, graph, optimizer_mlp, criterion, n_epochs=150)
#
# test_acc = eval_node_classifier(mlp, graph, graph.test_mask)
# print(f'Test Acc: {test_acc:.3f}')

from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = self.conv2(x, edge_index)

        return output


# gcn = GCN().to(device)
# optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
# criterion = nn.CrossEntropyLoss()
# gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)
#
# test_acc = eval_node_classifier(gcn, graph, graph.test_mask)
# print(f'Test Acc: {test_acc:.3f}')

# Link prediction is trickier than node classification as we need some tweaks to make predictions on edges using node
# embeddings. The prediction steps are described below:
#
# An encoder creates node embeddings by processing the graph with two convolution layers. We randomly add negative
# links to the original graph. This makes the model task binary classifications with the positive links from the
# original edges and the negative links from the added edges. A decoder makes link predictions (i.e. binary
# classifications) on all the edges including the negative links using node embeddings. It calculates a dot product
# of the node embeddings from pair of nodes on each edge. Then, it aggregates the values across the embedding
# dimension and creates a single value on every edge that represents the probability of edge existence.

from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def train_link_predictor(
        model, train_data, val_data, optimizer, criterion, n_epochs=100
):
    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")

    return model


@torch.no_grad()
def eval_link_predictor(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

#
# import torch_geometric.transforms as T
#
# split = T.RandomLinkSplit(
#     num_val=0.05,
#     num_test=0.1,
#     is_undirected=True,
#     add_negative_train_samples=False,
#     neg_sampling_ratio=1.0,
# )
# train_data, val_data, test_data = split(graph)
# model = Net(dataset.num_features, 128, 64).to(device)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()
# model = train_link_predictor(model, train_data, val_data, optimizer, criterion)
#
# test_auc = eval_link_predictor(model, test_data)
# print(f"Test: {test_auc:.3f}")

