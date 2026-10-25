# PyTorch Examples

Notebooks covering PyTorch fundamentals: tensors, neural networks, training loops, and image classification.
Each part has an Exercises notebook and a Solution notebook (Part 6 has a single notebook).

## Setup

There is no `environment.yml` here. Create a conda environment and install PyTorch and torchvision:

```bash
conda create -n pytorch python=3.10
conda activate pytorch
conda install pytorch torchvision -c pytorch
```

## Files

| File | Description |
|------|-------------|
| `dl_pytorch_Part 1 - Tensors in PyTorch (Exercises / Solution).ipynb` | Tensor operations and autograd basics |
| `dl_pytorch_Part 2 - Neural Networks in PyTorch (Exercises / Solution).ipynb` | Building a feedforward network |
| `dl_pytorch_Part 3 - Training Neural Networks (Exercises / Solution).ipynb` | Training loop with loss and optimizer |
| `dl_pytorch_Part 4 - Fashion-MNIST (Exercises / Solution).ipynb` | Fashion-MNIST classification |
| `dl_pytorch_Part 5 - Inference and Validation (Exercises / Solution).ipynb` | Inference and validation |
| `dl_pytorch_Part 6 - Saving and Loading Models.ipynb` | Model persistence |
| `dl_pytorch_Part 7 - Loading Image Data (Exercises / Solution).ipynb` | Loading image data |
| `dl_pytorch_Part 8 - Transfer Learning (Exercises / Solution).ipynb` | Transfer learning |
| `dl_pytorch_fc_model.py` | Reusable fully-connected model class |
| `dl_pytorch_helper.py` | Helper functions |
