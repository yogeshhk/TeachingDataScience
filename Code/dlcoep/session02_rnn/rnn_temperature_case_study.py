"""
DLCoEP Session 2 (RNN): the sensor-series case study of LaTeX/dl_rnn_temperature_case_study.tex.

  a real daily series -> sliding windows -> (batch, steps, features) -> a GRU and an LSTM
  -> compare with "same as yesterday" and a straight-line fit on the last 7 days.

Data: ../../ml/data/daily-minimum-temperatures-in-me.csv
      3,650 daily minimum temperatures (deg C), Melbourne, 1981-01-01 to 1990-12-31.
      29 February of 1984 and 1988 is absent from the file; the series is treated as consecutive days.

Run: conda activate genai && python rnn_temperature_case_study.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "ml", "data", "daily-minimum-temperatures-in-me.csv")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- The data ---------------------------------------------------------------
df = pd.read_csv(DATA)
series = df["Temperature"].to_numpy(dtype=np.float32)
dates = pd.to_datetime(df["Date"])
print("days:", len(series), " from", dates.iloc[0].date(), "to", dates.iloc[-1].date())
print("min %.1f  mean %.2f  max %.1f  std %.2f" % (series.min(), series.mean(), series.max(), series.std()))

# ---- Windows: 7 days in, the next day out -----------------------------------
STEPS = 7
n_train = int(0.8 * len(series))                      # chronological split, no shuffling across time
mu, sd = series[:n_train].mean(), series[:n_train].std()
z = (series - mu) / sd                                # scale with training statistics only
print("scaling: train mean %.2f, std %.2f" % (mu, sd))


def windows(a):
    x = np.stack([a[i:i + STEPS] for i in range(len(a) - STEPS)])
    y = a[STEPS:]
    return torch.tensor(x).unsqueeze(-1), torch.tensor(y)          # (N, STEPS, 1), (N,)


x_all, y_all = windows(z)
split = n_train - STEPS
xtr, ytr, xte, yte = x_all[:split], y_all[:split], x_all[split:], y_all[split:]
print("windows: train", tuple(xtr.shape), "test", tuple(xte.shape),
      " (batch, steps, features)")
print("test period: %s to %s" % (dates.iloc[n_train].date(), dates.iloc[-1].date()))


# ---- Baselines, in degrees C ------------------------------------------------
def to_c(a):
    return a * sd + mu


def mae_rmse(pred_c, true_c):
    e = pred_c - true_c
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


true_c = to_c(yte.numpy())
persist = to_c(xte[:, -1, 0].numpy())                              # same as yesterday
print("\nBaselines on the test period (deg C):")
m, r = mae_rmse(persist, true_c)
print("  same as yesterday: MAE %.2f  RMSE %.2f" % (m, r))
mean7 = to_c(xte[:, :, 0].numpy().mean(1))                         # mean of the last 7 days
m, r = mae_rmse(mean7, true_c)
print("  mean of last 7 days: MAE %.2f  RMSE %.2f" % (m, r))
from numpy.linalg import lstsq
Xtr_flat = np.c_[xtr[:, :, 0].numpy(), np.ones(len(xtr))]
coef = lstsq(Xtr_flat, ytr.numpy(), rcond=None)[0]                 # linear model on the 7 values
lin = to_c(np.c_[xte[:, :, 0].numpy(), np.ones(len(xte))] @ coef)
m, r = mae_rmse(lin, true_c)
print("  linear fit on 7 days: MAE %.2f  RMSE %.2f" % (m, r))


# ---- The recurrent models ---------------------------------------------------
class Net(nn.Module):
    def __init__(self, kind, hidden=16):
        super().__init__()
        self.rnn = (nn.GRU if kind == "GRU" else nn.LSTM)(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)                  # out: (batch, steps, hidden)
        return self.head(out[:, -1]).squeeze(-1)    # last step only


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fit(kind, epochs=30, lr=5e-3, batch=64):
    torch.manual_seed(SEED)
    net = Net(kind)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.mse_loss(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
        if ep in (1, 5, 10, 20, 30):
            net.eval()
            with torch.no_grad():
                mae = float(np.abs(to_c(net(xte).numpy()) - true_c).mean())
            print("  %s epoch %2d: test MAE %.2f" % (kind, ep, mae))
    net.eval()
    with torch.no_grad():
        return net, to_c(net(xte).numpy())


print("\nShapes: input (batch, steps, features) = (64, %d, 1); hidden state 16; output one number" % STEPS)
for kind in ("GRU", "LSTM"):
    print("Parameters, %s(1, 16) + linear head: %d" % (kind, n_params(Net(kind))))

results = {}
for kind in ("GRU", "LSTM"):
    print("\nTraining the %s (30 epochs, Adam, lr 0.005, batch 64):" % kind)
    net, pred = fit(kind)
    results[kind] = pred
    m, r = mae_rmse(pred, true_c)
    print("  %s test: MAE %.2f  RMSE %.2f" % (kind, m, r))

# ---- Where the error is -----------------------------------------------------
best = min(results, key=lambda k: mae_rmse(results[k], true_c)[0])
err = np.abs(results[best] - true_c)
print("\nBest model: %s. Share of test days within 1 deg C: %.1f%% (yesterday: %.1f%%)" % (
    best, 100 * (err <= 1).mean(), 100 * (np.abs(persist - true_c) <= 1).mean()))
print("Largest single-day error: %.1f deg C" % err.max())

# ---- Figure for the slides --------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = 90
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(true_c[:k], color="black", lw=1.4, label="actual")
    ax.plot(persist[:k], color="tab:orange", lw=1, ls="--", label="same as yesterday")
    ax.plot(results[best][:k], color="tab:blue", lw=1.2, label=best)
    ax.set_xlabel("day of the test period")
    ax.set_ylabel("min temp (deg C)")
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dlcoep_rnn_forecast.png"), dpi=150)
    plt.close(fig)
    print("Saved dlcoep_rnn_forecast.png in", OUT_DIR)
except ImportError:
    print("matplotlib not installed; figure skipped", file=sys.stderr)
