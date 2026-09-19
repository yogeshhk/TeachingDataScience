"""
Session 10 (Logistic Regression): the sigmoid plot from LaTeX/ml_logisticregression.tex.

The sigmoid squashes any real number z into (0, 1), which is why logistic regression can output a
probability. This is the only runnable code in the Session 10 slides.

Run: conda activate mlcoep && python sigmoid_plot.py
(On a machine without a display the figure is saved as sigmoid.png next to this script.)
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel(r'$\phi (z)$')

print("sigmoid(0)  =", sigmoid(0.0))
print("sigmoid(-7) = %.5f, sigmoid(7) = %.5f" % (sigmoid(-7), sigmoid(7)))

if plt.get_backend().lower() == 'agg':
    out = os.path.join(os.path.dirname(__file__), "sigmoid.png")
    plt.savefig(out)
    print("Saved", out)
else:
    plt.show()
