"""
Session 20 (ME Applications): SVM handwritten digit recognition, from LaTeX/ml_course_demo_svm_digits.tex.

  1. Load the built-in Digits data (1,797 images, 8 x 8 pixels = 64 features, 10 classes).
  2. Split 80/20, standardize, fit an RBF-kernel SVC (C=10, gamma=0.001).
  3. Print the classification report, plot the confusion matrix and the misclassified digits.

Result the slides quote: test accuracy 0.9806 (7 errors out of 360 test images).

Data: sklearn.datasets.load_digits (built in, no download).

Run: conda activate mlcoep && python demo_svm_digits.py
(On a machine without a display the figures are saved as PNG files next to this script.)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(__file__)
HEADLESS = plt.get_backend().lower() == 'agg'


def finish(name):
    """Save the current figure on a headless machine, otherwise show it."""
    if HEADLESS:
        out = os.path.join(HERE, name)
        plt.savefig(out)
        print("Saved", out)
        plt.close()
    else:
        plt.show()


# ---- Load and visualize ------------------------------------------------------
digits = load_digits()
print("Data shape:", digits.data.shape)   # (1797, 64)
print("Classes   :", digits.target_names)  # [0 1 2 ... 9]

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, img, label in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(img, cmap='gray_r')
    ax.set_title(str(label))
    ax.axis('off')
plt.suptitle("Sample Digits")
finish("digits_samples.png")

# ---- Train the SVM -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=10, gamma=0.001)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
print("Accuracy: %.4f  (%d errors of %d)" % (
    (y_pred == y_test).mean(), (y_pred != y_test).sum(), len(y_test)))

# ---- Confusion matrix --------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("SVM Digit Recognition: Confusion Matrix")
plt.tight_layout()
finish("digits_confusion_matrix.png")

# ---- Misclassified digits ----------------------------------------------------
wrong = np.where(y_pred != y_test)[0]
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax in axes.ravel():
    ax.axis('off')
for ax, idx in zip(axes.ravel(), wrong[:10]):
    ax.imshow(X_test[idx].reshape(8, 8), cmap='gray_r')
    ax.set_title(f"True:{y_test[idx]} Pred:{y_pred[idx]}")
plt.suptitle("Misclassified Digits")
plt.tight_layout()
finish("digits_misclassified.png")
