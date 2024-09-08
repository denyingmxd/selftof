import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Generate a tensor with integers from 1 to 64 and reshape it to 8x8
original_tensor = torch.arange(1, 65).view(1, 1, 8, 8).float()
down_sampled_tensor = F.adaptive_avg_pool2d(original_tensor, (8,8))
print(original_tensor.shape)
print(down_sampled_tensor.shape)

print((original_tensor == down_sampled_tensor).all())