# Keras / Deep Learning with Python

Notebooks from François Chollet's *Deep Learning with Python* (chapters 3–8), plus supplementary GAN examples.

## Setup
```bash
conda env create -f environment.yml
conda activate keras
```

## Contents by Chapter

| Chapter | Topic | Files |
|---------|-------|-------|
| 3 | Fundamentals | Movie reviews (binary classification), Newswires (multi-class), House prices (regression) |
| 4 | Regularization | Overfitting and underfitting strategies |
| 5 | CNNs | Convnets intro, small datasets with augmentation, pretrained VGG16, conv visualization |
| 6 | RNNs & Embeddings | One-hot encoding, word embeddings, simple RNN, LSTM, 1D convnets for sequences |
| 8 | Generative Models | Text generation (LSTM), DeepDream, neural style transfer, VAEs, GANs |

All files follow the naming convention: `dl_keras_<chapter>.<section>-<topic>_fchollet.ipynb`

## Source
François Chollet, *Deep Learning with Python*, Manning Publications.
Notebooks adapted from the book's official repository.
