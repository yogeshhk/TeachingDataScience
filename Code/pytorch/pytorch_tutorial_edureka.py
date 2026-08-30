# https://medium.com/edureka/pytorch-tutorial-9971d66f6893

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized
# range [-1, 1].

def get_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True,
                                             transform=transform)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True,
                                            transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                               shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random image
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    # show
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    return train_loader, test_loader


def imshow(img):
    img = img / 2 + 0.5  # un normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_network(model, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")


def evaluate_network(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    train_loader, test_loader = get_data()
    cifar_net = CIFAR10_Net()
    train_network(cifar_net, train_loader)
    evaluate_network(cifar_net, test_loader)
