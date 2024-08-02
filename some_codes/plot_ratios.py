import numpy as np

from PIL import Image as pil
import matplotlib.pyplot as plt
import os
import tqdm
import matplotlib
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines




test_files = readlines('/data/laiyan/codes/PLNet/splits/nyu/test_files.txt')
data_root = '/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/'

files = test_files

edge_crop = 16
# folder = 'mono_naive_ep_40_sm_0.1'
# folder = 'mono_real_rgbd_pose_rgbd_ep_40_sm_0.1_area_mean_area_L2_weighted_12_0.01_scale_depth_8'
folder = 'mono_real_rgbd_pose_rgbd_sparse_6_ep_40_sm_0.1_area_mean_area_L2_weighted_12_0.01_scale_depth_8'
# pred_root = '/data/laiyan/codes/PLNet/tmp/{}/models/weights_39/{}/pred_median/'.format(folder,folder)
pred_root = '/data/laiyan/codes/PLNet/tmp/{}/models/weights_39/{}/pred/'.format(folder,folder)

depth_gts = []
pred_depths = []
for i in tqdm.tqdm(range(len(files))[:]):
    index = files[i].split()[-1]
    scene = files[i].split()[0]
    depth_path = os.path.join(data_root, scene, index+ "_depth.png")
    depth_gt = pil.open(depth_path)
    depth_gt = depth_gt.crop((edge_crop, edge_crop, 640 - edge_crop, 480 - edge_crop))

    depth_gt = np.array(depth_gt).astype(np.float32) / 1000  # Convert to meters
    depth_gt = depth_gt.flatten()

    pred_depth = np.load(os.path.join(pred_root, index + "_pred_depth.npz"))['arr_0']
    pred_depth = pred_depth.flatten()

    depth_gts.append(depth_gt)
    pred_depths.append(pred_depth)

depth_gts = np.concatenate(depth_gts).flatten()
pred_depths = np.concatenate(pred_depths).flatten()
# plt.scatter(depth_gts, pred_depths, s=0.00001)

#draw y =x
plt.plot([0, 15], [0, 15], color='red')
# xlim = [4,15]
# ylim = [4,15]
# xlim = [0,4]
# ylim = [0,4]
xlim = [0,15]
ylim = [0,15]
plt.hist2d(depth_gts, pred_depths, bins=100, cmap='viridis',
           range=[xlim, ylim],norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e6),
           )
plt.colorbar(label='Count')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Histogram of (X,Y) Data')

plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

