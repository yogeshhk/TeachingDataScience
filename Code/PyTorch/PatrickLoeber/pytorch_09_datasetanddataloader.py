# https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9

# Note: for 100 samples, batch_size = 20, thus 100/20 = 5 iterations for 1 epoch
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt("data/wine/wine.csv",
                        delimiter=',',dtype=np.float32,
                        skiprows=1)
        self.x = torch.from_numpy(xy[:,1:]) # first column is y, so avoid it
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_samples = xy.shape[0]


    def __getitem__(self, item):
        # dataset[item]
        return self.x[item],self.y[item]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(f"Features {features} Labels {labels}")

dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)

dataiter = iter(dataloader)
features, labels = dataiter.next()
print(f"Features {features} Labels {labels}")

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

for epoch in range(num_epochs):
    for i,(inputs,labels) in enumerate(dataloader):
        if (i+1)%5==0:
            print(f"Epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations} inputs {inputs.shape}")

