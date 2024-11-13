#!/usr/bin/env python3

# Example ANN regression computed on GPU

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

# CUDA
device = torch.cuda.is_available()
print(device)
## Get Id of default device
if device == 'cuda':
    torch.cuda.current_device()
# 0
# Get name device with ID '0'
if device == 'cuda':
    torch.cuda.get_device_name(0)

# Returns the current GPU memory usage by
# tensors in bytes for a given device
if device == 'cuda':
    torch.cuda.memory_allocated()

# Returns the current GPU memory managed by the
# caching allocator in bytes for a given device
if device == 'cuda':
    torch.cuda.memory_cached()


# Data
X = torch.linspace(1,50,50).reshape(-1,1)
torch.manual_seed(71)
e = torch.randint(-8, 9, (50,1), dtype=torch.float)
y = 2*X + 1 + e

if device == 'cuda':
    X = torch.FloatTensor(X).cuda()
    y = torch.FloatTensor(y).cuda()

# model
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

torch.manual_seed(32)
model = Model(1, 1)

## Sending Models to GPU
if device == 'cuda':
    # From the discussions here: discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda
    next(model.parameters()).is_cuda
    model = model.cuda()
    next(model.parameters()).is_cuda

# test
w1,b1 = model.linear.weight.item(), model.linear.bias.item()
print(f'Initial weight: {w1:.8f}, Initial bias: {b1:.8f}')

x1 = np.array([X.min(),X.max()])
y1 = x1 * w1 + b1

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

# train the model
epochs = 100
losses = []
start = time.time()
for i in range(epochs):
    i+=1
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    losses.append(float(loss.detach().numpy()))
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}  weight: {model.linear.weight.item():10.8f} bias: {model.linear.bias.item():10.8f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'TOTAL TRAINING TIME: {time.time()-start}')

w1,b1 = model.linear.weight.item(), model.linear.bias.item()
print(f'Current weight: {w1:.8f}, Current bias: {b1:.8f}')
y1 = x1*w1 + b1
print(x1)
print(y1)

# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('epoch')

plt.scatter(X.numpy(), y.numpy())
plt.plot(x1,y1,'r')
plt.title('Current Model')
plt.ylabel('y')
plt.xlabel('x')
plt.show()

