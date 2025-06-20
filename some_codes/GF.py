import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import time
from PIL import Image

def upscale_depth(depth_map_lr, rgb_hr):
    """
    Upscale a low-resolution depth map using a high-resolution RGB image as a guide.

    Args:
    - depth_map_lr (np.array): Low-resolution depth map.
    - rgb_hr (np.array): High-resolution RGB image used as a guide.
    - scale (int): Scaling factor.

    Returns:
    - np.array: Upscaled depth map.
    """
    # Upscale the low-resolution depth map to the size of the high-resolution RGB image
    assert type(depth_map_lr) == np.ndarray, "depth_map_lr should be a numpy array"
    assert type(rgb_hr) == np.ndarray, "rgb_hr should be a numpy array"
    assert len(depth_map_lr.shape) == 2, "depth_map_lr should be a 2D array"
    assert len(rgb_hr.shape) == 3, "rgb_hr should be a 3D array"
    assert rgb_hr.shape[2] == 3, "rgb_hr should have 3 channels"


    height, width = rgb_hr.shape[:2]
    # depth_map_upscaled_init = cv2.resize(depth_map_lr, (width, height), interpolation=cv2.INTER_LINEAR)
    depth_map_upscaled_init = cv2.resize(depth_map_lr, (width, height), interpolation=cv2.INTER_NEAREST)

    # Apply the guided filter
    # Note: Experiment with the radius and eps values to get the best results for your specific scenario
    radius = 10  # Radius of the guided filter
    eps = 1.0   # Regularization parameter of the guided filter (a larger value smoothes more)

    depth_map_upscaled_refine = cv2.ximgproc.guidedFilter(guide=rgb_hr, src=depth_map_upscaled_init, radius=radius, eps=eps, dDepth=-1)
    return depth_map_upscaled_refine,depth_map_upscaled_init
def downsample_depth(depth_map_hr, scalex,scaley):
    """
    Downsample a high-resolution depth map.

    Args:
    - depth_map_hr (np.array): High-resolution depth map.
    - scale (int): Downscaling factor.

    Returns:
    - np.array: Downsampled depth map.
    """
    width = depth_map_hr.shape[1] // scalex
    height = depth_map_hr.shape[0] // scaley
    depth_map_lr = cv2.resize(depth_map_hr, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth_map_lr
# Load your images


if __name__ == '__main__':

    scalex = 256//8
    scaley = 256//8
    edge_crop = 16

    rgb_path = "/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/nyu2_test/00000_colors.png"
    depth_path = "/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/nyu2_test/00000_depth.png"

    depth_gt = Image.open(depth_path)
    depth_gt = depth_gt.crop((edge_crop, edge_crop, 640 - edge_crop, 480 - edge_crop))
    depth_gt = depth_gt.resize((256, 256))
    depth_gt = np.array(depth_gt).astype(np.float32) / 1000
    depth_gt = 1./depth_gt

    depth_map_lr = downsample_depth(depth_gt, scalex,scaley)
    # depth_map_lr[0,0] = 0
    # depth_map_lr[2,2] = 0
    rgb_hr = Image.open(rgb_path)
    #change to grayscale
    # rgb_hr = rgb_hr.convert('L')
    rgb_hr = rgb_hr.crop((edge_crop, edge_crop, 640 - edge_crop, 480 - edge_crop))
    rgb_hr = rgb_hr.resize((256, 256))
    rgb_hr = np.array(rgb_hr).astype(np.float32) / 255

    # Assume the scaling factor is known


    # Perform depth super-resolution
    upscaled_depth_map_refine,upscaled_depth_map_init = upscale_depth(depth_map_lr, rgb_hr)

    plt.imshow(rgb_hr); plt.show()
    # plt.imshow(depth_map_lr); plt.show()
    # plt.imshow(upscaled_depth_map_init); plt.show()
    plt.imshow(upscaled_depth_map_refine); plt.show()
    plt.imshow(depth_gt); plt.show()