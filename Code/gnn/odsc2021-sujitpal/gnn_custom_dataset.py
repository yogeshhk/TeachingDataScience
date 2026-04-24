"""
GNN Custom Dataset: Synthetic Random Graphs with PyG InMemoryDataset
Demonstrates building a custom PyTorch Geometric Dataset from raw CSV files.
Generates 50 Barabasi-Albert random graphs with random node features and labels.
Source: ODSC West 2021 tutorial by Sujit Pal (solution notebook 07)
"""
import os
import shutil
import numpy as np
import torch
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data
from torch_geometric.loader import DataLoader

DATA_ROOT = os.path.join('data', 'random_graphs')


# ── Data generation ───────────────────────────────────────────────────────────

def generate_raw_data(raw_dir, num_graphs=50, num_nodes=100, num_features=10):
    """Generate random node-feature CSVs and edge-list CSVs."""
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir)
    print(f'Generating {num_graphs} random graphs in {raw_dir}...')
    for i in range(num_graphs):
        edge_index    = pyg_utils.barabasi_albert_graph(num_nodes, 50)
        node_features = torch.rand((num_nodes, num_features), dtype=torch.float32)

        node_path = os.path.join(raw_dir, f'node-{i}.csv')
        with open(node_path, 'w') as f:
            for j in range(num_nodes):
                feats = ','.join(f'{v:.5f}' for v in node_features[j].tolist())
                f.write(f'{j}\t{feats}\n')

        edge_path = os.path.join(raw_dir, f'edge-{i}.csv')
        with open(edge_path, 'w') as f:
            for j in range(edge_index.size(1)):
                f.write(f'{edge_index[0, j]}\t{edge_index[1, j]}\n')
    print(f'Generated {num_graphs} graphs.')


# ── Custom Dataset ─────────────────────────────────────────────────────────────

class RandomGraphDataset(pyg_data.Dataset):
    """
    Custom PyG Dataset that reads node-feature and edge CSVs,
    builds Data objects, and caches processed tensors to disk.
    """

    def __init__(self, root, num_graphs=50, transform=None, pre_transform=None):
        self.num_graphs = num_graphs
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        names = []
        for i in range(self.num_graphs):
            names.append(f'node-{i}.csv')
            names.append(f'edge-{i}.csv')
        return names

    @property
    def processed_file_names(self):
        return [f'random-{i}.pt' for i in range(self.num_graphs)]

    def download(self):
        pass   # raw data generated externally via generate_raw_data()

    def process(self):
        print('Processing raw CSVs...')
        for i in range(self.num_graphs):
            features = []
            with open(os.path.join(self.raw_dir, f'node-{i}.csv')) as f:
                for line in f:
                    _, feats = line.strip().split('\t')
                    features.append([float(x) for x in feats.split(',')])
            x = torch.tensor(np.array(features), dtype=torch.float32)

            edges = []
            with open(os.path.join(self.raw_dir, f'edge-{i}.csv')) as f:
                for line in f:
                    src, dst = line.strip().split('\t')
                    edges.append((int(src), int(dst)))
                    edges.append((int(dst), int(src)))   # make undirected
            edge_index = torch.tensor(edges).t().to(torch.long)

            y    = torch.randint(low=0, high=2, size=(1,))
            data = pyg_data.Data(x=x, edge_index=edge_index, y=y)
            torch.save(data, os.path.join(self.processed_dir, f'random-{i}.pt'))
        print('Processing done.')

    def len(self):
        return self.num_graphs

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'random-{idx}.pt'),
                          weights_only=False)


if __name__ == '__main__':
    # Generate raw CSV files
    raw_dir = os.path.join(DATA_ROOT, 'raw')
    generate_raw_data(raw_dir, num_graphs=50, num_nodes=100, num_features=10)

    # Remove cached processed data so it's always rebuilt from raw
    processed_dir = os.path.join(DATA_ROOT, 'processed')
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    # Build dataset
    dataset = RandomGraphDataset(root=DATA_ROOT)
    print(f'\nDataset: {dataset}')
    print(f'  Graphs: {len(dataset)}')
    print(f'  Features per node: {dataset.num_features}')
    print(f'  First graph: {dataset[0]}')

    # DataLoader smoke test
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in loader:
        print(f'\nFirst batch: {batch}')
        print(f'  Batch size (graphs): {batch.num_graphs}')
        print(f'  Total nodes in batch: {batch.x.shape[0]}')
        break

    print('\nDone.')
