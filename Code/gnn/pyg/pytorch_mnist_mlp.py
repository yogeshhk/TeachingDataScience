"""
PyTorch Basics: MNIST Digit Classification with MLP
Feedforward neural network on the MNIST handwritten digits dataset.
"""
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

os.makedirs('data', exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

INPUT_SIZE = 784  # 28x28
HIDDEN_SIZE = 500
NUM_CLASSES = 10
NUM_EPOCHS = 2
BATCH_SIZE = 100
LR = 0.001

train_dataset = torchvision.datasets.MNIST(root='data', train=True,
                                            transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='data', train=False,
                                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


model = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f'\nTraining MLP on MNIST ({NUM_EPOCHS} epochs)...')
n_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 100 == 0:
            print(f'  Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{n_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    correct = 0
    total = len(test_loader.dataset)
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
    print(f'\nTest Accuracy: {100 * correct / total:.2f}%  ({correct}/{total})')
