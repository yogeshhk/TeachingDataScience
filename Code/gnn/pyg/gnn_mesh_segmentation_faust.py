"""
GNN 3D Mesh Segmentation: FAUST Human Body Dataset with FeastNet
Vertex-level segmentation of 3D human body meshes into 12 body-part classes.

Dataset setup (manual step required):
  The MPI-FAUST dataset must be downloaded and placed at data/MPI-FAUST.zip
  Download from: http://faust.is.tue.mpg.de/ (requires free registration)
  Alternatively, if you have the gdown link:
    pip install gdown
    python -c "import gdown; gdown.download(id='1CvfkR6iFOpfo0yRyaVOvgv2piphP6pze', output='data/MPI-FAUST.zip')"
"""
import os
import sys
from time import sleep
from pathlib import Path
from itertools import tee
from functools import lru_cache

import numpy as np
from tqdm import tqdm
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.transforms import BaseTransform, Compose, FaceToEdge
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.loader import DataLoader

os.makedirs('data', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'data'
print(f'Using device: {DEVICE}')


def load_mesh(mesh_filename: Path):
    mesh = trimesh.load_mesh(str(mesh_filename), process=False)
    vertices = torch.from_numpy(mesh.vertices).to(torch.float)
    faces = torch.from_numpy(mesh.faces).t().to(torch.long).contiguous()
    return vertices, faces


class SegmentationFaust(InMemoryDataset):
    map_seg_label_to_id = {
        'head': 0, 'torso': 1, 'left_arm': 2, 'left_hand': 3,
        'right_arm': 4, 'right_hand': 5, 'left_upper_leg': 6,
        'left_lower_leg': 7, 'left_foot': 8, 'right_upper_leg': 9,
        'right_lower_leg': 10, 'right_foot': 11,
    }

    def __init__(self, root, train=True, pre_transform=None):
        super().__init__(root, pre_transform=pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    @lru_cache(maxsize=32)
    def _segmentation_labels(self):
        path = Path(self.root) / 'MPI-FAUST' / 'segmentations.npz'
        seg_labels = np.load(str(path))['segmentation_labels']
        return torch.from_numpy(seg_labels).to(torch.int64)

    def _mesh_filenames(self):
        return sorted((Path(self.root) / 'MPI-FAUST' / 'meshes').glob('*.ply'))

    def process(self):
        zip_path = Path(self.root) / 'MPI-FAUST.zip'
        if not zip_path.exists():
            print(f'ERROR: {zip_path} not found. See module docstring for download instructions.')
            sys.exit(1)
        extract_zip(str(zip_path), self.root, log=False)

        data_list = []
        for mesh_filename in self._mesh_filenames():
            vertices, faces = load_mesh(mesh_filename)
            data = Data(x=vertices, face=faces)
            data.segmentation_labels = self._segmentation_labels
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])


class NormalizeUnitSphere(BaseTransform):
    def __call__(self, data: Data):
        if data.x is not None:
            x = data.x - data.x.mean(dim=0)
            data.x = x / x.norm(dim=1).max()
        return data

    def __repr__(self):
        return 'NormalizeUnitSphere()'


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_mlp_layers(channels, activation, output_activation=nn.Identity):
    layers = []
    *intermediate, final = pairwise(channels)
    for in_ch, out_ch in intermediate:
        layers += [nn.Linear(in_ch, out_ch), activation()]
    layers += [nn.Linear(*final), output_activation()]
    return nn.Sequential(*layers)


class FeatureSteeredConvolution(MessagePassing):
    """FeastNet: Feature-steered graph convolutions (Verma et al., CVPR 2018)."""

    def __init__(self, in_channels, out_channels, num_heads,
                 ensure_trans_invar=True, bias=True, with_self_loops=True):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.with_self_loops = with_self_loops
        self.linear = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.u = nn.Linear(in_channels, num_heads, bias=False)
        self.c = nn.Parameter(torch.Tensor(num_heads))
        self.v = None if ensure_trans_invar else nn.Linear(in_channels, num_heads, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.linear.weight)
        nn.init.uniform_(self.u.weight)
        nn.init.normal_(self.c, mean=0.0, std=0.1)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=0.1)
        if self.v is not None:
            nn.init.uniform_(self.v.weight)

    def forward(self, x, edge_index):
        if self.with_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.shape[0])
        out = self.propagate(edge_index, x=x)
        return out if self.bias is None else out + self.bias

    def _attention(self, x_i, x_j):
        logits = self.u(x_i - x_j) + self.c if self.v is None else self.u(x_i) + self.v(x_j) + self.c
        return F.softmax(logits, dim=1)

    def message(self, x_i, x_j):
        attn = self._attention(x_i, x_j)
        x_j = self.linear(x_j).view(-1, self.num_heads, self.out_channels)
        return (attn.view(-1, self.num_heads, 1) * x_j).sum(dim=1)


class GraphFeatureEncoder(nn.Module):
    def __init__(self, in_features, conv_channels, num_heads,
                 apply_batch_norm=True, ensure_trans_invar=True, bias=True, with_self_loops=True):
        super().__init__()
        conv_params = dict(num_heads=num_heads, ensure_trans_invar=ensure_trans_invar,
                           bias=bias, with_self_loops=with_self_loops)
        self.apply_batch_norm = apply_batch_norm
        all_channels = [in_features] + conv_channels
        self.conv_layers = nn.ModuleList([
            FeatureSteeredConvolution(in_ch, out_ch, **conv_params)
            for in_ch, out_ch in pairwise(all_channels)
        ])
        *intermediate_ch, _ = conv_channels
        self.batch_layers = nn.ModuleList([nn.BatchNorm1d(ch) for ch in intermediate_ch]) \
            if apply_batch_norm else [None] * len(intermediate_ch)

    def forward(self, x, edge_index):
        *first_convs, last_conv = self.conv_layers
        for conv, bn in zip(first_convs, self.batch_layers):
            x = conv(x, edge_index)
            x = F.relu(x)
            if bn is not None:
                x = bn(x)
        return last_conv(x, edge_index)


class MeshSeg(nn.Module):
    def __init__(self, in_features, encoder_features, conv_channels,
                 encoder_channels, decoder_channels, num_classes, num_heads, apply_batch_norm=True):
        super().__init__()
        self.input_encoder = get_mlp_layers([in_features] + encoder_channels, nn.ReLU)
        self.gnn = GraphFeatureEncoder(encoder_features, conv_channels, num_heads, apply_batch_norm)
        *_, final_ch = conv_channels
        self.final_projection = get_mlp_layers([final_ch] + decoder_channels + [num_classes], nn.ReLU)

    def forward(self, data):
        x = self.input_encoder(data.x)
        x = self.gnn(x, data.edge_index)
        return self.final_projection(x)


def train_epoch(net, loader, optimizer, loss_fn):
    net.train()
    total = 0.0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out, data.segmentation_labels.squeeze())
        loss.backward()
        total += loss.item()
        optimizer.step()
    return total / len(loader)


@torch.no_grad()
def accuracy(preds, labels):
    return float((preds.argmax(dim=-1, keepdim=True) == labels).sum()) / labels.shape[0]


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    accs = []
    for data in loader:
        data = data.to(DEVICE)
        accs.append(accuracy(net(data), data.segmentation_labels))
    return sum(accs) / len(accs)


if __name__ == '__main__':
    pre_transform = Compose([FaceToEdge(remove_faces=False), NormalizeUnitSphere()])

    try:
        train_data = SegmentationFaust(root=DATA_ROOT, pre_transform=pre_transform)
        test_data = SegmentationFaust(root=DATA_ROOT, train=False, pre_transform=pre_transform)
    except SystemExit:
        print('\nExiting: please download the FAUST dataset first (see module docstring).')
        sys.exit(1)

    train_loader = DataLoader(train_data, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)
    print(f'Train meshes: {len(train_data)}, Test meshes: {len(test_data)}')

    model_params = dict(
        in_features=3, encoder_features=16,
        conv_channels=[32, 64, 128, 64], encoder_channels=[16],
        decoder_channels=[32], num_classes=12, num_heads=12, apply_batch_norm=True,
    )
    net = MeshSeg(**model_params).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 50
    best_test_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    print(f'\nTraining FeastNet for {num_epochs} epochs...')
    with tqdm(range(num_epochs), unit='epoch') as t:
        for epoch in t:
            loss = train_epoch(net, train_loader, optimizer, loss_fn)
            train_acc = evaluate(net, train_loader)
            test_acc = evaluate(net, test_loader)
            t.set_postfix(loss=f'{loss:.4f}', train_acc=f'{100*train_acc:.1f}%',
                          test_acc=f'{100*test_acc:.1f}%')
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(net.state_dict(), 'checkpoints/mesh_seg_best.pt')
            sleep(0.05)

    print(f'\nBest Test Accuracy: {best_test_acc:.4f}')
    print('Checkpoint saved to checkpoints/mesh_seg_best.pt')
