# https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=10

# Note: for 100 samples, batch_size = 20, thus 100/20 = 5 iterations for 1 epoch
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt("data/wine/wine.csv",
                        delimiter=',',dtype=np.float32,
                        skiprows=1)
        self.x = xy[:,1:] # first column is y, so avoid it
        self.y = xy[:,[0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, item):
        # dataset[item]
        sample = self.x[item],self.y[item]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self,sample):
        input, labels = sample
        return torch.from_numpy(input), torch.from_numpy(labels)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self,sample):
        input, labels = sample
        input *= self.factor
        return input, labels

print('Without Transform')
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)