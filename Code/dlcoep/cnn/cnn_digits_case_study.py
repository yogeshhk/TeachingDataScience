"""
DLCoEP, CNN: the handwritten-digits case study of LaTeX/dl_cnn_digits_case_study.tex.

  shapes through a small CNN -> parameter count against a fully connected net -> train on MNIST
  -> test accuracy, the digits that get confused, the learned 3x3 filters, and a few mistakes.

Data: MNIST through torchvision (one download of about 11 MB into ../data, gitignored).
If there is no network, it falls back to scikit-learn's bundled 8x8 digits (1,797 images), so it
still runs offline; the numbers on the slides are the MNIST ones.

Run: conda activate genai && python cnn_digits_case_study.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)


# ---- The data ---------------------------------------------------------------
def load_mnist():
    from torchvision import datasets
    train = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)
    xtr = train.data.float().div(255).unsqueeze(1)      # (60000, 1, 28, 28)
    xte = test.data.float().div(255).unsqueeze(1)
    return xtr, train.targets, xte, test.targets, "MNIST"


def load_sklearn_digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    x = torch.tensor(d.images, dtype=torch.float32).div(16).unsqueeze(1)   # (1797, 1, 8, 8)
    y = torch.tensor(d.target)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)
    return xtr, ytr, xte, yte, "sklearn digits (offline fallback)"


try:
    xtr, ytr, xte, yte, source = load_mnist()
except Exception as exc:                      # no network, or torchvision missing
    print("MNIST not available (%s); using sklearn digits" % type(exc).__name__)
    xtr, ytr, xte, yte, source = load_sklearn_digits()

side = xtr.shape[-1]
print("Data:", source)
print("train", tuple(xtr.shape), "test", tuple(xte.shape),
      "pixel range %.1f to %.1f" % (xtr.min().item(), xtr.max().item()))
print("classes:", sorted(set(ytr.tolist())), " images per class in train:",
      np.bincount(ytr.numpy()).tolist())


# ---- The model --------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, side):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)        # (1, S, S) -> (8, S-2, S-2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)       # -> (16, ., .)
        s = ((side - 2) // 2 - 2) // 2
        self.fc = nn.Linear(16 * s * s, 10)

    def forward(self, x, trace=False):
        shapes = [("input", tuple(x.shape[1:]))]
        x = F.relu(self.conv1(x)); shapes.append(("conv1 + relu", tuple(x.shape[1:])))
        x = F.max_pool2d(x, 2);    shapes.append(("max pool 2x2", tuple(x.shape[1:])))
        x = F.relu(self.conv2(x)); shapes.append(("conv2 + relu", tuple(x.shape[1:])))
        x = F.max_pool2d(x, 2);    shapes.append(("max pool 2x2", tuple(x.shape[1:])))
        x = x.flatten(1);          shapes.append(("flatten", tuple(x.shape[1:])))
        x = self.fc(x);            shapes.append(("fully connected", tuple(x.shape[1:])))
        return (x, shapes) if trace else x


model = SmallCNN(side)
_, shapes = model(xtr[:1], trace=True)
print("\nShapes through the network (one image):")
for name, shp in shapes:
    print("  %-16s %s" % (name, shp))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


mlp = nn.Sequential(nn.Flatten(), nn.Linear(side * side, 128), nn.ReLU(), nn.Linear(128, 10))
print("\nParameters: CNN %d, a 128-unit fully connected net %d" % (n_params(model), n_params(mlp)))
for name, p in model.named_parameters():
    print("  %-14s %-18s %d" % (name, tuple(p.shape), p.numel()))


# ---- Train ------------------------------------------------------------------
def accuracy(m, x, y):
    m.eval()
    with torch.no_grad():
        return (m(x).argmax(1) == y).float().mean().item()


def train(m, epochs, lr=1e-3, batch=128):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(m(xtr[idx]), ytr[idx]).backward()
            opt.step()
        print("  epoch %d: test accuracy %.4f" % (ep, accuracy(m, xte, yte)))


EPOCHS = 3 if source == "MNIST" else 15
print("\nTraining the CNN (%d epochs, Adam, batch 128):" % EPOCHS)
train(model, EPOCHS)
cnn_acc = accuracy(model, xte, yte)

print("\nTraining the fully connected net, same data and epochs:")
train(mlp, EPOCHS)
mlp_acc = accuracy(mlp, xte, yte)
print("\nTest accuracy: CNN %.4f, fully connected %.4f" % (cnn_acc, mlp_acc))


# ---- Where it goes wrong ----------------------------------------------------
model.eval()
with torch.no_grad():
    pred = model(xte).argmax(1)
conf = np.zeros((10, 10), dtype=int)
for t, p in zip(yte.numpy(), pred.numpy()):
    conf[t, p] += 1
per_class = conf.diagonal() / conf.sum(1)
print("\nPer-digit accuracy:", {d: round(float(a), 3) for d, a in enumerate(per_class)})
off = [(conf[i, j], i, j) for i in range(10) for j in range(10) if i != j]
off.sort(reverse=True)
print("Most confused (true -> predicted, count):", [(i, j, int(c)) for c, i, j in off[:4]])
wrong = int((pred != yte).sum())
print("Wrong: %d of %d test images" % (wrong, len(yte)))


# ---- Figures for the slides -------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = model.conv1.weight.detach().numpy()[:, 0]           # (8, 3, 3)
    fig, axes = plt.subplots(1, 8, figsize=(8, 1.3))
    for k, ax in enumerate(axes):
        ax.imshow(w[k], cmap="RdBu", vmin=-abs(w).max(), vmax=abs(w).max())
        ax.set_title("f%d" % (k + 1), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dlcoep_cnn_filters.png"), dpi=150)
    plt.close(fig)

    bad = np.where((pred != yte).numpy())[0][:8]
    fig, axes = plt.subplots(1, 8, figsize=(8, 1.5))
    for ax, i in zip(axes, bad):
        ax.imshow(xte[i, 0].numpy(), cmap="gray")
        ax.set_title("%d as %d" % (yte[i].item(), pred[i].item()), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dlcoep_cnn_mistakes.png"), dpi=150)
    plt.close(fig)
    print("\nSaved dlcoep_cnn_filters.png and dlcoep_cnn_mistakes.png in", OUT_DIR)
except ImportError:
    print("matplotlib not installed; figures skipped", file=sys.stderr)
