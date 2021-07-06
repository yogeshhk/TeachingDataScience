# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=2
# Installation
#
# https://pytorch.org/get-started/locally/
#
#
# Install Cuda Toolkit
# Development environment for creating high performance GPU-accelerated applications
# You need an NVIDIA GPU in your machine:
#
# https://developer.nvidia.com/cuda-downloads
#
# Legacy releases
# 10.1 update 2
# select OS (e.g. Windows 10)
#
# Download and install
#
# # Create conda environment and activate it
# conda create -n pytorch python=3.7
# conda activate pytorch
#
# # Install pytorch
# conda install pytorch torchvision -c pytorch
# or with GPU
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Verification:
import torch
x = torch.rand(3)
print(x)
print(torch.__version__)
print(torch.cuda.is_available())
