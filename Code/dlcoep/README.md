# DLCoEP: Deep Learning for Mechanical Engineers (CoEP)

Code behind the two deep-learning lectures that follow the MLCoEP course: one on convolutional
neural networks and one on recurrent neural networks. Each lecture pairs the generic slides
(`LaTeX/dl_cnn.tex`, `LaTeX/dl_rnn.tex`) with one hands-on case study.

| Topic | Slides | Script | Data |
|---|---|---|---|
| CNN | `LaTeX/dl_cnn_digits_case_study.tex` | `cnn/cnn_digits_case_study.py` | MNIST handwritten digits, downloaded once through torchvision (about 11 MB, into `data/`, not committed). With no network the script falls back to scikit-learn's bundled 8 x 8 digits. |
| RNN | `LaTeX/dl_rnn_temperature_case_study.tex` | `rnn/rnn_temperature_case_study.py` | `Code/ml/data/daily-minimum-temperatures-in-me.csv`, 3,650 daily minimum temperatures, 1981 to 1990. No download. |
| CNN, concept frames | "Why Not Just a Dense Layer?" and "Filters by Numbers" frames in `LaTeX/dl_cnn.tex` | `cnn/cnn_filters_by_hand.py` | Parameter counts, a pixel-shuffle test and a translation test, Sobel and kernel worked numbers, pooling, and a synthetic machined-plate figure. MNIST is read from `data/` (already downloaded by the case-study script); the rest is synthetic. |
| RNN, concept frames | "Why Not Just a Dense Layer for Sequences?", "One Step by Hand", and "Fading by Numbers" frames in `LaTeX/dl_rnn.tex`; the readout and attention frames in `LaTeX/dl_rnn_readouts.tex` | `rnn/rnn_sequence_demos.py` | Synthetic sequences only: parameter counts, an order test, a position test, gradient fading, an early-evidence task with several readouts, and a tiny attention example. |

## Run

```bash
conda env create -f environment.yml
conda activate dlcoep
python cnn/cnn_digits_case_study.py
python rnn/rnn_temperature_case_study.py
```

Each script prints the numbers quoted on the slides and writes its figures to `out/` (not
committed). The figures used by the slides are copied to `LaTeX/images/` as `dlcoep_*.png`.
Seeds are fixed and training runs on the CPU, so a rerun on the same setup repeats the numbers
(the slides were produced with Python 3.11 and PyTorch 2.7.1; other versions can differ in the
last digit).

## Notes

- The case studies compare against simple baselines; the point is the comparison, not a win for
  the network.
- The temperature series is not machine data. It stands in for a regularly sampled sensor signal.
  Swap in a vibration or temperature log from a machine by changing `DATA` and the column name.
- The file skips 29 February of 1984 and 1988; the scripts treat the series as consecutive days.
