"""
GNN Point Cloud Classification: GeometricShapes Dataset with PointNet
Custom message-passing (PointNet-style) for classifying 3D geometric shapes.
"""
import os
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, SamplePoints, RandomRotate
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_cluster import knn_graph

os.makedirs('data/GeometricShapes', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


class PointNetLayer(MessagePassing):
    """Custom message-passing layer: learns local geometry from spatial position differences."""

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = Sequential(
            Linear(in_channels + 3, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_j, pos_i):
        # Encode spatial relation (pos_j - pos_i) combined with neighbor features
        spatial = pos_j - pos_i
        inp = torch.cat([h_j, spatial], dim=-1) if h_j is not None else spatial
        return self.mlp(inp)


class PointNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, num_classes)

    def forward(self, pos, batch):
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = global_max_pool(h, batch)
        return self.classifier(h)


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        logits = model(data.pos, data.batch)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(DEVICE)
        logits = model(data.pos, data.batch)
        pred = logits.argmax(dim=-1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


if __name__ == '__main__':
    dataset = GeometricShapes(root='data/GeometricShapes')
    print(f'Dataset: GeometricShapes')
    print(f'  Graphs: {len(dataset)}, Classes: {dataset.num_classes}')

    train_dataset = GeometricShapes(root='data/GeometricShapes', train=True,
                                    transform=SamplePoints(128))
    test_dataset = GeometricShapes(root='data/GeometricShapes', train=False,
                                   transform=SamplePoints(128))

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10)

    model = PointNet(num_classes=dataset.num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print(f'\nTraining PointNet (no augmentation)...')
    for epoch in range(1, 51):
        loss = train(model, train_loader, optimizer, criterion)
        test_acc = test(model, test_loader)
        if epoch % 10 == 0:
            print(f'  Epoch {epoch:02d}: Loss={loss:.4f}, Test Acc={test_acc:.4f}')

    # Test with random rotation augmentation (harder)
    random_rotate = Compose([
        RandomRotate(degrees=180, axis=0),
        RandomRotate(degrees=180, axis=1),
        RandomRotate(degrees=180, axis=2),
    ])
    rotated_test = GeometricShapes(root='data/GeometricShapes', train=False,
                                   transform=Compose([random_rotate, SamplePoints(128)]))
    rotated_loader = DataLoader(rotated_test, batch_size=10)
    rot_acc = test(model, rotated_loader)
    print(f'\nTest Accuracy (with random rotation): {rot_acc:.4f}')
    print('Done.')
