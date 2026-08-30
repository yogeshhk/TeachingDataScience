"""
PyTorch Basics: CIFAR-10 Image Classification with CNN
Convolutional neural network on the CIFAR-10 dataset (32x32 RGB images, 10 classes).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

os.makedirs('data', exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)    # 32x32 -> 30x30
        self.pool = nn.MaxPool2d(2, 2)       # 30x30 -> 15x15
        self.conv2 = nn.Conv2d(32, 64, 3)   # 15x15 -> 13x13
        self.conv3 = nn.Conv2d(64, 64, 3)   # 13x13 -> 11x11 -> pool -> 6x6 (after pool on conv2)
        # After conv1+pool: 32x15x15; after conv2+pool: 64x6x6; after conv3: 64x4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 32 x 15 x 15
        x = self.pool(F.relu(self.conv2(x)))   # -> 64 x 6 x 6
        x = F.relu(self.conv3(x))              # -> 64 x 4 x 4
        x = torch.flatten(x, 1)               # -> 1024
        x = F.relu(self.fc1(x))               # -> 64
        return self.fc2(x)                     # -> 10


model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f'\nTraining CNN on CIFAR-10 ({NUM_EPOCHS} epochs)...')
n_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 100 == 0:
            print(f'  Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{n_steps}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    correct = 0
    total = len(test_loader.dataset)
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
    print(f'\nTest Accuracy: {100 * correct / total:.2f}%  ({correct}/{total})')
