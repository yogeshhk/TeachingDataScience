# DLCoEP: Deep Learning for Mechanical Engineers (CoEP)

Code behind the two deep-learning sessions that follow the MLCoEP course: one on convolutional
neural networks and one on recurrent neural networks. Each session pairs the generic slides
(`LaTeX/dl_cnn.tex`, `LaTeX/dl_rnn.tex`) with one hands-on case study.

| Session | Slides | Script | Data |
|---|---|---|---|
| 1. CNN | `LaTeX/dl_cnn_digits_case_study.tex` | `session01_cnn/cnn_digits_case_study.py` | MNIST handwritten digits, downloaded once through torchvision (about 11 MB, into `data/`, not committed). With no network the script falls back to scikit-learn's bundled 8 x 8 digits. |
| 2. RNN | `LaTeX/dl_rnn_temperature_case_study.tex` | `session02_rnn/rnn_temperature_case_study.py` | `Code/ml/data/daily-minimum-temperatures-in-me.csv`, 3,650 daily minimum temperatures, 1981 to 1990. No download. |
| 1. CNN, concept frames | "Why Not Just a Dense Layer?" and "Filters by Numbers" frames in `LaTeX/dl_cnn.tex` | `session01_cnn/cnn_filters_by_hand.py` | Parameter counts, a pixel-shuffle test and a translation test, Sobel and kernel worked numbers, pooling, and a synthetic machined-plate figure. MNIST is read from `data/` (already downloaded by the case-study script); the rest is synthetic. |
| 2. RNN, concept frames | "Why Not Just a Dense Layer for Sequences?", "One Step by Hand", readout and attention frames in `LaTeX/dl_rnn.tex` | `session02_rnn/rnn_sequence_demos.py` | Synthetic sequences only: parameter counts, an order test, a position test, gradient fading, an early-evidence task with several readouts, and a tiny attention example. |

## Run

```bash
conda env create -f environment.yml
conda activate dlcoep
python session01_cnn/cnn_digits_case_study.py
python session02_rnn/rnn_temperature_case_study.py
```

Each script prints the numbers quoted on the slides and writes its figures to `out/` (not
committed). The figures used by the slides are copied to `LaTeX/images/` as `dlcoep_*.png`.

## What the numbers are

Seeds are fixed and training runs on the CPU, so a rerun on the same setup repeats the numbers.
The slides were produced with Python 3.11 and PyTorch 2.7.1; other versions can differ in the
last digit.

Session 1, MNIST, 3 epochs, Adam, batch 128:

| Model | Parameters | Test accuracy |
|---|---|---|
| Fully connected 784-128-10 | 101,770 | 96.26% |
| Small CNN (8 and 16 filters of 3 x 3) | 5,258 | 97.70% |

Session 2, one-step-ahead forecast on the 730 test days (1989 and 1990):

| Method | MAE (deg C) | RMSE (deg C) |
|---|---|---|
| Same as yesterday | 1.95 | 2.48 |
| Mean of the last 7 days | 2.04 | 2.60 |
| Linear fit on the last 7 days | 1.76 | 2.24 |
| GRU, 16 units | 1.75 | 2.22 |
| LSTM, 16 units | 1.74 | 2.21 |

The recurrent nets beat "same as yesterday" and tie the linear fit. The slides say so; the case
study is about comparing against simple baselines, not about a win for the RNN.

## Notes

- The temperature series is not machine data. It stands in for a regularly sampled sensor signal.
  Swap in a vibration or temperature log from a machine by changing `DATA` and the column name.
- The file skips 29 February of 1984 and 1988; the scripts treat the series as consecutive days.
