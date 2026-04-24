# Install Cuda 12.1.1, nvidia-smi shows I can install upto CUDA 12.7
# CudaNN is of 12.x of dec 5, 2023
# My GPU is Nvidia GeForce MX570 A
# https://www.youtube.com/watch?v=nATRPPZ5dGE
# https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning?tab=readme-ov-file

import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print("\nGPU Device Details:")
if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")