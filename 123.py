import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Step 1: Generate a random binary mask of shape (9, 12) containing only 0 and 1
mask = torch.randint(0, 2, (1, 1, 9, 12)).float().requires_grad_(True) # Adding batch and channel dimensions

# Step 2: Upsample the mask to shape (144, 192)
upsampled_mask = F.interpolate(mask, size=(144, 192), mode='nearest')

downsampled_mask = F.interpolate(upsampled_mask, size=(9, 12), mode='nearest')
downsampled_mask.sum().backward()
print(mask.grad)
# print(downsampled_mask.grad)