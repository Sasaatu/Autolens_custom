import torch

x = torch.randn(1024, 1024, device="cuda")
for _ in range(10000):
    y = torch.matmul(x, x)

print('Process is finished.')