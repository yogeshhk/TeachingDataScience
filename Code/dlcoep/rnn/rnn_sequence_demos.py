"""
DLCoEP, RNN: the numbers behind the "Why not a dense layer for sequences?", "One step by
hand", "One vector per sequence" and "Attention" frames of LaTeX/dl_rnn.tex.

  A. parameter counts: dense layer on a fixed window against recurrent layers
  B. one recurrence step by hand, and why order changes the answer
  C. order test: a bag of symbols cannot tell "A before B" from "B before A"; a dense net on
     positions can, but only at one fixed length
  D. position test: train with the spike early in the sequence, test with it late
  E. gradient fading: how much the first input still moves the last hidden state, by length
  F. early-evidence task: last state, mean pooling, max pooling, bidirectional
  G. attention on three hidden states, with numbers

Run: conda activate genai && python rnn_sequence_demos.py
Seeds are fixed and training runs on the CPU. Everything is synthetic, so no data download.
"""
import math
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
torch.set_num_threads(4)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def fit(model, x, y, epochs, lr=0.01, batch=64, clip=None, seed=SEED):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    return model


# ---- A. parameter counts ----------------------------------------------------
print("=== A. Parameter counts (T steps, d features per step, H hidden units) ===")
d, H = 8, 32
print("d=%d, H=%d" % (d, H))
print("%8s %22s %12s %12s %12s" % ("T", "dense on T*d inputs", "RNN", "GRU", "LSTM"))
for T in (10, 100, 1000):
    dense = n_params(nn.Linear(T * d, H))
    print("%8d %22d %12d %12d %12d" % (T, dense, n_params(nn.RNN(d, H)), n_params(nn.GRU(d, H)),
                                       n_params(nn.LSTM(d, H))))
print("RNN formula: d*H + H*H + 2*H = %d  (PyTorch keeps two bias vectors)" % (d * H + H * H + 2 * H))
print("The recurrent counts do not depend on T. The dense layer only accepts one fixed T.")


# ---- B. one recurrence step by hand -----------------------------------------
print("\n=== B. One recurrence step by hand: h_t = tanh(wx*x_t + wh*h_(t-1)), wx=1.0, wh=0.5, h_0=0 ===")


def run(seq, wx=1.0, wh=0.5):
    h, hs = 0.0, []
    for x in seq:
        h = math.tanh(wx * x + wh * h)
        hs.append(h)
    return hs


for seq in ([1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]):
    print("x = %-10s h = %s   final %.4f" % (seq, ["%.4f" % v for v in run(seq)], run(seq)[-1]))
print("Same inputs, different order, different final state: the state remembers order.")
print("Only the last-step input is fresh: the 1 at step 1 is down to %.4f after two more steps."
      % run([1, 0, 0])[-1])


# ---- C. order test ----------------------------------------------------------
print("\n=== C. Order test: label = does symbol A come before symbol B? ===")
V = 4  # symbols 0 and 3 are filler; 1 is A and 2 is B, each exactly once


def order_data(n, T, rng):
    x = rng.choice([0, 3], size=(n, T))
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        a, b = rng.choice(T, 2, replace=False)
        x[i, a], x[i, b] = 1, 2
        y[i] = int(a < b)
    return torch.tensor(x), torch.tensor(y)


class BagDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(V, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(F.one_hot(x, V).sum(1).float())


class FlatDense(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(T * V, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(F.one_hot(x, V).flatten(1).float())


class GRULast(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.rnn = nn.GRU(V, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        _, h = self.rnn(F.one_hot(x, V).float())
        return self.fc(h[-1])


rng = np.random.default_rng(SEED)
xtr, ytr = order_data(4000, 6, rng)
x6, y6 = order_data(1000, 6, rng)
x10, y10 = order_data(1000, 10, rng)
print("%-34s %8s %12s %14s" % ("model (trained on length 6)", "params", "test len 6", "test len 10"))
for name, make in (("bag of symbols (counts) + dense", BagDense), ("dense on the 6 positions", lambda: FlatDense(6)),
                   ("GRU, last state", GRULast)):
    torch.manual_seed(SEED)
    m = fit(make(), xtr, ytr, 15, batch=128)
    try:
        longer = "%.1f%%" % (100 * accuracy(m, x10, y10))
    except RuntimeError:
        longer = "cannot run"
    print("%-34s %8d %11.1f%% %14s" % (name, n_params(m), 100 * accuracy(m, x6, y6), longer))

# The same GRU, trained on a mix of lengths, tested on seen and unseen lengths.
mixed = {L: order_data(1500, L, rng) for L in (6, 8, 10, 12)}
torch.manual_seed(SEED)
gru = GRULast()
opt = torch.optim.Adam(gru.parameters(), lr=0.01)
g = torch.Generator().manual_seed(SEED)
for _ in range(15):
    for L, (xm, ym) in mixed.items():
        perm = torch.randperm(len(xm), generator=g)
        for i in range(0, len(xm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            F.cross_entropy(gru(xm[idx]), ym[idx]).backward()
            opt.step()
print("GRU trained on lengths 6, 8, 10, 12 (same parameters, %d):" % n_params(gru))
for L in (6, 10, 14, 20):
    xt_, yt_ = order_data(1000, L, rng)
    print("   test length %2d%s: %.1f%%" % (L, "" if L in mixed else " (never seen)", 100 * accuracy(gru, xt_, yt_)))


# ---- D. position test -------------------------------------------------------
print("\n=== D. Position test: label = is there a spike? Trained with the spike in steps 1-10 of 20 ===")
T = 20


def spike_data(n, lo, hi, rng):
    x = rng.normal(0, 0.1, (n, T, 1)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n // 2):
        x[i, rng.integers(lo, hi), 0] += 1.0
        y[i] = 1
    p = rng.permutation(n)
    return torch.tensor(x[p]), torch.tensor(y[p])


rng = np.random.default_rng(SEED)
xs, ys = spike_data(2000, 0, 10, rng)
xs_early, ys_early = spike_data(1000, 0, 10, rng)
xs_late, ys_late = spike_data(1000, 10, 20, rng)


class DenseSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(T, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


class GRUSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 8, batch_first=True)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])


print("%-22s %8s %16s %16s" % ("model", "params", "spike in 1-10", "spike in 11-20"))
for name, make in (("dense 20-32-2", DenseSeq), ("GRU, 8 units", GRUSeq)):
    torch.manual_seed(SEED)
    m = fit(make(), xs, ys, 30, lr=0.01, batch=64)
    print("%-22s %8d %15.1f%% %15.1f%%" % (name, n_params(m), 100 * accuracy(m, xs_early, ys_early),
                                           100 * accuracy(m, xs_late, ys_late)))


# ---- E. gradient fading -----------------------------------------------------
print("\n=== E. Gradient fading: |d(last hidden state) / d(first input)|, untrained, H=32, median of 20 seeds ===")
print("%6s %14s %14s %14s" % ("T", "tanh RNN", "GRU", "LSTM"))
for T_ in (5, 10, 20, 40, 80):
    row = []
    for cls in (nn.RNN, nn.GRU, nn.LSTM):
        vals = []
        for s in range(20):
            torch.manual_seed(s)
            m = cls(1, 32, batch_first=True)
            x = torch.randn(1, T_, 1, requires_grad=True)
            out, _ = m(x)
            out[0, -1].sum().backward()
            vals.append(x.grad[0, 0, 0].abs().item())
        row.append(statistics.median(vals))
    print("%6d %14.2e %14.2e %14.2e" % (T_, *row))


# ---- F. early-evidence task -------------------------------------------------
print("\n=== F. Early-evidence task: label = is there a +3 marker in steps 1-3 of 60? (steps 4-60 are noise) ===")
T = 60


def early_data(n, rng):
    x = rng.normal(0, 1, (n, T, 1)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n // 2):
        x[i, rng.integers(0, 3), 0] = 3.0
        y[i] = 1
    p = rng.permutation(n)
    return torch.tensor(x[p]), torch.tensor(y[p])


class Reader(nn.Module):
    def __init__(self, cell, readout, bidir=False, hidden=16):
        super().__init__()
        self.rnn = cell(1, hidden, batch_first=True, bidirectional=bidir)
        self.readout = readout
        self.fc = nn.Linear(hidden * (2 if bidir else 1), 2)

    def forward(self, x):
        out, h = self.rnn(x)
        if isinstance(h, tuple):          # LSTM returns (h, c)
            h = h[0]
        if self.readout == "last":
            z = out[:, -1]                # last step of the output (forward direction only for bidir=False)
        elif self.readout == "mean":
            z = out.mean(1)
        elif self.readout == "max":
            z = out.amax(1)
        else:                             # "final": last state of every direction, concatenated
            z = torch.cat([h[i] for i in range(h.shape[0])], dim=1)
        return self.fc(z)


rng = np.random.default_rng(SEED)
xe, ye = early_data(2000, rng)
xe_t, ye_t = early_data(1000, rng)
configs = [
    ("tanh RNN, last state", nn.RNN, "last", False, 20, 0.01),
    ("tanh RNN, mean over time", nn.RNN, "mean", False, 20, 0.01),
    ("tanh RNN, max over time", nn.RNN, "max", False, 20, 0.01),
    ("tanh RNN, bidirectional, final states", nn.RNN, "final", True, 20, 0.01),
    ("GRU, last state, 20 epochs", nn.GRU, "last", False, 20, 0.01),
    ("GRU, last state, 100 epochs, lr 0.003", nn.GRU, "last", False, 100, 0.003),
    ("LSTM, last state, 100 epochs, lr 0.003", nn.LSTM, "last", False, 100, 0.003),
    ("GRU, max over time, 20 epochs", nn.GRU, "max", False, 20, 0.01),
    ("GRU, bidirectional, final states", nn.GRU, "final", True, 20, 0.01),
]
print("%-40s %8s %12s %12s" % ("model", "params", "train acc", "test acc"))
for name, cell, readout, bidir, epochs, lr in configs:
    torch.manual_seed(SEED)
    m = fit(Reader(cell, readout, bidir), xe, ye, epochs, lr=lr, batch=64, clip=1.0)
    print("%-40s %8d %11.1f%% %11.1f%%" % (name, n_params(m), 100 * accuracy(m, xe, ye),
                                          100 * accuracy(m, xe_t, ye_t)))


# ---- G. attention by hand ---------------------------------------------------
print("\n=== G. Attention on three hidden states ===")
hs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
q = torch.tensor([1.0, 0.0])
scores = hs @ q
weights = torch.softmax(scores, dim=0)
context = weights @ hs
print("hidden states h1, h2, h3:", hs.tolist(), "  query:", q.tolist())
print("scores (dot products):", scores.tolist())
print("weights (softmax):", ["%.4f" % w for w in weights.tolist()], " sum %.4f" % weights.sum().item())
print("context = weighted sum:", ["%.4f" % v for v in context.tolist()])
print("Plain average of the three states, for comparison:", ["%.4f" % v for v in hs.mean(0).tolist()])
