# PyTorch Examples

Scripts and notebooks covering PyTorch fundamentals: tensors, neural networks, training loops, and image classification.

## Setup

```bash
conda create -n pytorch python=3.10
conda activate pytorch
conda install pytorch torchvision -c pytorch
```

## Files

| File | Description |
|------|-------------|
| `dl_pytorch_tensors.ipynb` | Tensor operations and autograd basics |
| `dl_pytorch_neural_network.ipynb` | Building a simple feedforward network |
| `dl_pytorch_training.ipynb` | Training loop with loss, optimizer, and validation |
| `dl_pytorch_fashionmnist_*.ipynb` | Fashion-MNIST classification series |
| `dl_pytorch_fc_model.py` | Reusable fully-connected model class |
| `dl_pytorch_helper.py` | Utility functions (plot, load, save) |

## Learning Path

1. Tensors → autograd
2. `nn.Module` and layer definitions
3. Training loop (forward, loss, backward, optimizer step)
4. Inference and model persistence (`torch.save` / `torch.load`)
5. Full classification pipeline on Fashion-MNIST
