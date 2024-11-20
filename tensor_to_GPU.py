#!/usr/bin/env python3

# Example moving PyTorch tensors to GPU

import torch
import numpy as np

if torch.cuda.is_available():
    points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
else:
    print('CUDA is not available')

# to
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

if torch.cuda.is_available():
    points_gpu = points.to(device='cuda')
else:
    print('CUDA is not available')

# specify the number of the GPU device
if torch.cuda.is_available():
    points_gpu = points.to(device='cuda:0')
else:
    print('CUDA is not available')


if torch.cuda.is_available():
    points = 2 * points  # <1> on CPU
    points_gpu = 2 * points.to(device='cuda')  # <2> on GPU
else:
    print('CUDA is not available')

if torch.cuda.is_available():
    points_gpu = points_gpu + 4

if torch.cuda.is_available():
    points_cpu = points_gpu.to(device='cpu')

points_gpu = points.cuda()  # <1>
print(points_gpu)
points_gpu = points.cuda(0)
print(points_gpu)
points_cpu = points_gpu.cpu()
print(points_cpu)
