"""
GNN Custom Message Passing Layer: Multi-head Attention (GAT-style)
Implements a custom MessagePassing layer from scratch using PyG primitives.
Demonstrates how to build custom graph convolution with max aggregation.
Source: ODSC West 2021 tutorial by Sujit Pal (solution notebook 06)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_scatter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


class MyCustomGATLayer(pyg_nn.MessagePassing):
    """
    Custom multi-head attention message-passing layer.
    - Learns separate linear projections for source (i) and target (j) nodes
    - Computes attention weights via leaky-ReLU gated scores
    - Aggregates messages using max (demonstrating scatter-based custom aggregation)
    """

    def __init__(self, input_dim, output_dim, heads=2, negative_slope=0.2,
                 dropout=0.0, **kwargs):
        super().__init__(node_dim=0, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_i = nn.Linear(input_dim, output_dim)
        self.lin_j = nn.Linear(input_dim, output_dim)

        F_per_head = output_dim // heads
        self.att_i = nn.Parameter(torch.rand(F_per_head, F_per_head))
        self.att_j = nn.Parameter(torch.rand(F_per_head, F_per_head))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_i.weight)
        nn.init.xavier_uniform_(self.lin_j.weight)
        nn.init.xavier_uniform_(self.att_i)
        nn.init.xavier_uniform_(self.att_j)

    def forward(self, x, edge_index):
        H, C = self.heads, self.output_dim

        x_i = self.lin_i(x).view(x.size(0), H, C // H)
        x_j = self.lin_j(x).view(x.size(0), H, C // H)

        alpha_i = torch.matmul(x_i, self.att_i)
        alpha_j = torch.matmul(x_j, self.att_j)

        out = self.propagate(edge_index,
                             num_nodes=x.size(0),
                             alpha=(alpha_i, alpha_j),
                             x=(x_i, x_j))
        return out.view(-1, C)

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        attention = F.leaky_relu(x_j * alpha_i + x_j * alpha_j,
                                 negative_slope=self.negative_slope)
        attention = pyg_utils.softmax(attention, index=index, ptr=ptr)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        return attention * x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(inputs, index, dim=0, reduce='max')


if __name__ == '__main__':
    torch.manual_seed(42)

    # Random graph: 20 nodes, 4-dim features
    num_nodes  = 20
    input_dim  = 4
    output_dim = 10  # must be divisible by heads=2

    x          = torch.rand((num_nodes, input_dim)).to(DEVICE)
    edge_index = pyg_utils.barabasi_albert_graph(num_nodes, 2).to(DEVICE)

    print(f'Input  x:          {tuple(x.shape)}')
    print(f'edge_index:        {tuple(edge_index.shape)}')

    layer = MyCustomGATLayer(input_dim, output_dim, heads=2).to(DEVICE)
    out   = layer(x, edge_index)

    print(f'Output (after GAT): {tuple(out.shape)}')
    print(f'\nLayer parameters:')
    for name, param in layer.named_parameters():
        print(f'  {name}: {tuple(param.shape)}')

    # Gradient smoke-test
    loss = out.sum()
    loss.backward()
    print('\nBackward pass OK - gradients computed successfully.')
