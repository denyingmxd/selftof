import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Generate a tensor with integers from 1 to 64 and reshape it to 8x8
original_tensor = torch.arange(1, 65).view(1, 1, 8, 8).float()

# Upsample the tensor
upsampled_tensor = F.interpolate(original_tensor,(9,12), mode='bilinear', align_corners=False)


# Function to visualize tensor
def visualize_tensor(tensor, title):
    array = tensor.squeeze().numpy()
    plt.imshow(array, cmap='viridis', interpolation='none')
    plt.colorbar()

    # Annotate with numbers
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            plt.text(j, i, str(array[i, j]), ha='center', va='center', color='white')

    plt.title(title)
    plt.show()


# Visualize the original and upsampled tensors
visualize_tensor(original_tensor, "Original Tensor")
visualize_tensor(upsampled_tensor, "Upsampled Tensor")