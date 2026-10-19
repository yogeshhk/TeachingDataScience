"""
DLCoEP Session 1 (CNN): the numbers behind the "Why not a dense layer?" and "Filters by numbers"
frames of LaTeX/dl_cnn.tex.

  A. parameter counts, dense layer against convolution
  B. the pixel-shuffle test on MNIST: a dense net does not care where pixels are, a CNN does
  C. the translation test: train with the feature on the left, test with it on the right
  D. filters by hand: Sobel on small patches and grids, four classic kernels, pooling
  E. a figure: Sobel edges of a synthetic machined plate

Run: conda activate genai && python cnn_filters_by_hand.py
MNIST is read from ../data (already downloaded by cnn_digits_case_study.py). Part B is skipped if
it is missing. Seeds are fixed and training runs on the CPU.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import convolve2d, correlate2d

SEED = 0
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)
np.set_printoptions(linewidth=140)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def show(title, arr):
    print(title)
    print(np.array2string(np.asarray(arr), separator=" ", formatter={"float_kind": lambda v: "%.2f" % v}))


# ---- A. parameter counts ----------------------------------------------------
print("=== A. Parameter counts ===")
h, w, c, hidden = 256, 256, 3, 1000
inputs = h * w * c
dense_params = inputs * hidden + hidden
print("Colour image %dx%dx%d = %d inputs; dense layer to %d units: %d weights + biases"
      % (h, w, c, inputs, hidden, dense_params))
print("  torch check:", n_params(nn.Linear(inputs, hidden)))
conv = nn.Conv2d(3, 32, kernel_size=3)
print("Conv layer, 32 filters of 3x3x3: %d parameters (%d weights + %d biases), for any image size"
      % (n_params(conv), conv.weight.numel(), conv.bias.numel()))
print("  ratio dense / conv = %.0f" % (n_params(nn.Linear(inputs, hidden)) / n_params(conv)))
mlp_mnist = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
print("MNIST 28x28: dense 784-128-10 = %d parameters" % n_params(mlp_mnist))
print("One 3x3 filter on a grey image: 9 weights + 1 bias = 10, on a 28x28 or a 2800x2800 image")


# ---- B. the pixel-shuffle test ----------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.flatten(1))


def fit(model, xtr, ytr, xte, yte, epochs, lr=1e-3, batch=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return (model(xte).argmax(1) == yte).float().mean().item()


print("\n=== B. Pixel-shuffle test (MNIST, 3 epochs, Adam, batch 128) ===")
try:
    from torchvision import datasets
    tr = datasets.MNIST(DATA_DIR, train=True, download=False)
    te = datasets.MNIST(DATA_DIR, train=False, download=False)
    xtr = tr.data.float().div(255).unsqueeze(1)
    xte = te.data.float().div(255).unsqueeze(1)
    ytr, yte = tr.targets, te.targets
    shuffle = torch.randperm(784, generator=torch.Generator().manual_seed(123))
    xtr_s = xtr.flatten(1)[:, shuffle].view(-1, 1, 28, 28)
    xte_s = xte.flatten(1)[:, shuffle].view(-1, 1, 28, 28)
    for name, make in (("dense 784-128-10", lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 128),
                                                                  nn.ReLU(), nn.Linear(128, 10))),
                       ("small CNN", SmallCNN)):
        torch.manual_seed(SEED)
        a = fit(make(), xtr, ytr, xte, yte, 3)
        torch.manual_seed(SEED)
        b = fit(make(), xtr_s, ytr, xte_s, yte, 3)
        print("%-18s normal images %.2f%%   same fixed shuffle of the pixel positions %.2f%%  (change %+.2f points)"
              % (name, 100 * a, 100 * b, 100 * (b - a)))
except Exception as exc:
    print("Part B skipped: %s: %s" % (type(exc).__name__, exc))


# ---- C. the translation test ------------------------------------------------
print("\n=== C. Translation test: feature seen only on the left half in training ===")
S = 12


def blob_images(n, side, rng):
    """Half the images carry a plus-shaped 3x3 blob (label 1), half are noise only (label 0).
    side='left' puts the blob centre in columns 1..4, side='right' in columns 7..10."""
    x = rng.normal(0, 0.1, (n, 1, S, S)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    cols = range(1, 5) if side == "left" else range(7, 11)
    for i in range(n // 2):
        r, cc = rng.integers(1, S - 1), rng.choice(list(cols))
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            x[i, 0, r + dr, cc + dc] += 1.0
        y[i] = 1
    p = rng.permutation(n)
    return torch.tensor(x[p]), torch.tensor(y[p])


rng = np.random.default_rng(SEED)
xl, yl = blob_images(2000, "left", rng)
xl_test, yl_test = blob_images(1000, "left", rng)
xr_test, yr_test = blob_images(1000, "right", rng)


class ConvGlobalMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(F.relu(self.conv(x)).amax(dim=(2, 3)))


class ConvFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.fc = nn.Linear(4 * (S - 2) * (S - 2), 2)

    def forward(self, x):
        return self.fc(F.relu(self.conv(x)).flatten(1))


def acc(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


models = {
    "dense 144-32-2": lambda: nn.Sequential(nn.Flatten(), nn.Linear(S * S, 32), nn.ReLU(), nn.Linear(32, 2)),
    "conv 3x3 + flatten + linear": ConvFlatten,
    "conv 3x3 + global max pool": ConvGlobalMax,
}
print("%-30s %8s  %12s %12s" % ("model", "params", "test: left", "test: right"))
for name, make in models.items():
    torch.manual_seed(SEED)
    m = make()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for _ in range(60):
        opt.zero_grad()
        F.cross_entropy(m(xl), yl).backward()
        opt.step()
    print("%-30s %8d  %11.1f%% %11.1f%%" % (name, n_params(m), 100 * acc(m, xl_test, yl_test),
                                            100 * acc(m, xr_test, yr_test)))


# ---- D. filters by hand -----------------------------------------------------
print("\n=== D. Filters by hand ===")
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)     # responds to left-right change
SY = SX.T.copy()                                                      # responds to top-bottom change
print("Sobel x (vertical edges):\n", SX.astype(int))
print("Sobel y (horizontal edges):\n", SY.astype(int))

patches = {
    "uniform 255": np.full((3, 3), 255.0),
    "dark left, bright right": np.array([[0, 0, 255]] * 3, dtype=float),
    "bright left, dark right": np.array([[255, 0, 0]] * 3, dtype=float),
    "dark top, bright bottom": np.array([[0] * 3, [0] * 3, [255] * 3], dtype=float),
}
print("\n%-26s %8s %8s" % ("3x3 patch", "Gx", "Gy"))
for name, p in patches.items():
    print("%-26s %8.0f %8.0f" % (name, (p * SX).sum(), (p * SY).sum()))

step = np.zeros((6, 6)); step[:, 3:] = 9
show("\nStep image 6x6 (0 left, 9 right):", step)
gx, gy = correlate2d(step, SX, "valid"), correlate2d(step, SY, "valid")
show("Sobel x on it (4x4):", gx)
show("Sobel y on it (4x4):", gy)

block = np.zeros((6, 6)); block[1:4, 1:4] = 9
gxb, gyb = correlate2d(block, SX, "valid"), correlate2d(block, SY, "valid")
show("\nBlock image 6x6 (3x3 block of 9):", block)
show("Sobel x (4x4):", gxb)
show("Sobel y (4x4):", gyb)
show("Magnitude sqrt(gx^2+gy^2) (4x4), rounded:", np.round(np.hypot(gxb, gyb)))

kernels = {
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "blur (box)": np.full((3, 3), 1 / 9),
    "edge (Laplacian)": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
}
spot = np.full((3, 3), 10.0); spot[1, 1] = 50.0
flat = np.full((3, 3), 10.0)
print("\nSame two patches through four kernels (one output number each):")
print("  patch A: 10 everywhere, 50 in the centre;  patch B: 10 everywhere")
print("%-18s %10s %10s" % ("kernel", "patch A", "patch B"))
for name, k in kernels.items():
    print("%-18s %10.2f %10.2f" % (name, (spot * k).sum(), (flat * k).sum()))

img = np.array([[1, 2, 0], [0, 1, 3], [4, 0, 1]], dtype=float)
k = np.array([[1, 0], [0, -1]], dtype=float)
t_conv = F.conv2d(torch.tensor(img)[None, None], torch.tensor(k)[None, None])[0, 0].numpy()
print("\nCross-correlation versus true convolution (kernel flipped):")
print("  torch conv2d equals scipy correlate2d:", bool(np.allclose(t_conv, correlate2d(img, k, 'valid'))))
print("  torch conv2d equals scipy convolve2d: ", bool(np.allclose(t_conv, convolve2d(img, k, 'valid'))))
print("  with Sobel x on the step image, convolve2d gives the negative of correlate2d:",
      bool(np.allclose(convolve2d(step, SX, 'valid'), -correlate2d(step, SX, 'valid'))))

A = np.array([[1, 9, 2, 0], [3, 4, 0, 1], [0, 2, 7, 5], [1, 0, 6, 8]], dtype=float)


def maxpool(a):
    return a.reshape(2, 2, 2, 2).max(axis=(1, 3))


A_in = A.copy(); A_in[0, 0], A_in[0, 1] = A[0, 1], A[0, 0]            # 9 moves one pixel, inside its window
A_cross = A.copy(); A_cross[0, 1], A_cross[0, 2] = A[0, 2], A[0, 1]   # 9 moves one pixel, across the window edge
A_shift = np.zeros_like(A); A_shift[:, 1:] = A[:, :-1]                # whole image moves right by one pixel
show("\nMax pooling 2x2 stride 2 on a 4x4 input:", A)
show("  pooled:", maxpool(A))
show("  9 moves one pixel inside its window, pooled:", maxpool(A_in))
show("  9 moves one pixel across the window edge, pooled:", maxpool(A_cross))
show("  whole image shifted right by 1 pixel, pooled:", maxpool(A_shift))
print("  entries that differ from the original pooled output after the whole-image shift: %d of 4"
      % int((maxpool(A_shift) != maxpool(A)).sum()))


# ---- E. figure --------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 96
    yy, xx = np.mgrid[0:n, 0:n]
    plate = np.full((n, n), 200.0)
    plate[:8, :] = plate[-8:, :] = plate[:, :8] = plate[:, -8:] = 60.0          # dark border
    plate[(yy - 48) ** 2 + (xx - 40) ** 2 < 14 ** 2] = 40.0                      # drilled hole
    plate[np.abs((yy - 20) - 0.5 * (xx - 55)) < 1.0] = 90.0                      # a thin scratch
    plate[:, 70:] = np.maximum(plate[:, 70:] - 20, 0)                            # lighting step
    plate += np.random.default_rng(SEED).normal(0, 3, plate.shape)
    gxp, gyp = correlate2d(plate, SX, "valid"), correlate2d(plate, SY, "valid")
    mag = np.hypot(gxp, gyp)
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
    for ax, im, ttl, cm in ((axes[0], plate, "plate with hole and scratch", "gray"),
                            (axes[1], gxp, "Sobel x: vertical edges", "RdBu"),
                            (axes[2], gyp, "Sobel y: horizontal edges", "RdBu"),
                            (axes[3], mag, "edge strength", "gray")):
        lim = np.abs(im).max()
        ax.imshow(im, cmap=cm, vmin=(-lim if cm == "RdBu" else None), vmax=(lim if cm == "RdBu" else None))
        ax.set_title(ttl, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dlcoep_cnn_sobel.png"), dpi=150)
    plt.close(fig)
    print("\nSaved dlcoep_cnn_sobel.png in", OUT_DIR)
except ImportError:
    print("matplotlib not installed; figure skipped", file=sys.stderr)
