import timm
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm
from PIL import Image as pil
import matplotlib.pyplot as plt
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines



split='test'
fpath = os.path.join(os.path.dirname(__file__), "../splits", 'nyu', "{}_files.txt")
base_dir = '/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/'
filenames = readlines(fpath.format(split))
histograms = []
edge_crop=16

histograms = []
bin_edges = None
depth_range = (0.01, 10)
num_bins = 100

for idx, file in enumerate(filenames):
    folder, frame_index = file.split()
    depth_path = os.path.join(base_dir, folder, str(frame_index) + "_depth.png")

    if not os.path.exists(depth_path):
        print('not exist')
        raise ValueError('not exist')

    depth_gt = pil.open(depth_path)
    depth_gt = depth_gt.crop((edge_crop, edge_crop, 640 - edge_crop, 480 - edge_crop))

    depth_gt = np.array(depth_gt).astype(np.float32) / 1000  # Convert to meters
    depth_array = depth_gt.flatten()

    # Compute histogram
    hist, edges = np.histogram(depth_array, bins=num_bins, range=depth_range)
    histograms.append(hist)

    if bin_edges is None:
        bin_edges = edges

# Convert list of histograms to numpy array and sum
total_hist = np.sum(histograms, axis=0)

# Calculate CDF
cdf = np.cumsum(total_hist)
cdf = cdf / cdf[-1]  # Normalize CDF

# Plot histogram
plt.figure(figsize=(12, 6))
plt.bar(bin_edges[:-1], total_hist, width=np.diff(bin_edges), align="edge", alpha=0.7)
plt.xlabel('Depth (meters)')
plt.ylabel('Frequency')
plt.title('Combined Depth Histogram for All Frames')
plt.xlim(depth_range)
plt.grid(True, alpha=0.3)
plt.show()

# Plot histogram and CDF together
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(bin_edges[:-1], total_hist, width=np.diff(bin_edges), align="edge", alpha=0.7, color='b')
ax1.set_xlabel('Depth (meters)')
ax1.set_ylabel('Frequency', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xlim(depth_range)

ax2 = ax1.twinx()
ax2.plot(bin_edges[:-1], cdf, color='r', linewidth=2)
ax2.set_ylabel('Cumulative Probability', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(0, 1)

plt.title('Combined Depth Histogram and CDF for All Frames')
plt.grid(True, alpha=0.3)
plt.show()