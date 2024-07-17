import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from matplotlib.patches import Rectangle
# data_path = "/data/laiyan/codes/calibrated-backprojection-network/data/ZJUL5-SLAM/Office/Raw/1656939229.205839.h5"
# data_path = "/data/laiyan/codes/calibrated-backprojection-network/data/ZJUL5-SLAM/Sofa2/Raw/1656936995.864139.h5"
# # data_path = "/data/laiyan/codes/deltar/data/ZJUL5/cafe2/1646380932.28328.h5"
# f = h5py.File(data_path, 'r')
# rgb = f['rgb'][:] #480,640,3
# depth = f['depth'][:]#480,640
# fr = f['fr'][:]#64,4
# hist_data = f['hist_data'][:] #64,2
# mask = f['mask'][:]#64,
# plt.imshow(rgb);plt.show()
# plt.imshow(depth);plt.show()
# print(mask.sum())
# plt.imshow(rgb);
# for i, xxx in enumerate(fr):
#     a, b, c, d = xxx
#     if mask[i]:
#         plt.gca().add_patch(Rectangle((b, a), c - a, d - b,
#                                   edgecolor='red',
#                                   facecolor='none',
#                                   lw=4))
# plt.show()

dirs = ['Sofa2','LivingRoom','Sofa','Office','Office2','Reception','Kitchen']
root = "/data/laiyan/codes/calibrated-backprojection-network/data/ZJUL5-SLAM/"
depth_hist = []
for dir in dirs:
    data_path = os.path.join(root,dir,'Raw')
    files = os.listdir(data_path)
    files.sort()
    for file in files:
        f = h5py.File(os.path.join(data_path,file), 'r')
        rgb = f['rgb'][:] #480,640,3
        depth = f['depth'][:]#480,640
        fr = f['fr'][:]#64,4
        hist_data = f['hist_data'][:] #64,2
        mask = f['mask'][:]#64,
        plt.imshow(rgb);
        print(mask.sum())
        for i, xxx in enumerate(fr):
            a, b, c, d = xxx
            if mask[i]:
                plt.gca().add_patch(Rectangle((b, a), c - a, d - b,
                                          edgecolor='red',
                                          facecolor='none',
                                          lw=4))
        plt.show()
        # break
    # break


import tqdm
# dirs = ['Sofa2','LivingRoom','Sofa','Office','Office2','Reception','Kitchen']
# root = "/data/laiyan/codes/calibrated-backprojection-network/data/ZJUL5-SLAM/"

# dirs = ["library2", "lab2", "cafe2", "teaching_region", "lab1", "cafe1", "library1", "classroom", "supermarket", "dorm", "dinner_room2", "dinner_room1", "showroom", "theater", "leisure_area"]
# root = "/data/laiyan/codes/deltar/data/ZJUL5/"
# depth_histograms = {}
# mask_histograms = {}
#
# # Loop through each directory
# for dir in dirs:
#     data_path = os.path.join(root, dir)
#     files = os.listdir(data_path)
#     depth_hist = []
#     mask_hist = []
#
#     # Loop through each file in the directory
#     for file in tqdm.tqdm(files, desc=f'Processing {dir}'):
#         if not file.endswith('.h5'):
#             continue
#         with h5py.File(os.path.join(data_path, file), 'r') as f:
#             if 'depth' not in f.keys() or 'mask' not in f.keys():
#                 continue
#             # print(os.path.join(data_path, file), 'r')
#             # rgb = f['rgb'][:]  # 480,640,3
#             depth = f['depth'][:]  # 480,640
#             # fr = f['fr'][:]  # 64,4
#             # hist_data = f['hist_data'][:]  # 64,2
#             mask = f['mask'][:]  # 64,
#
#             # Calculate histogram for the depth data
#             local_hist = np.histogram(depth.flatten(), bins=100, range=(0, 10))
#             depth_hist.append(local_hist[0])
#
#             # Calculate histogram for the mask data
#
#             mask_hist.append(mask.sum()/(len(mask)))
#
#     # Sum histograms for all files in the directory
#     if depth_hist:
#         total_hist = np.sum(depth_hist, axis=0)
#         depth_histograms[dir] = total_hist
#     if mask_hist:
#         mask_histograms[dir] = mask_hist
#
#     # break
#
# ncols = 4
#
# # Mask Histograms
# nrows = (len(mask_histograms) + ncols - 1) // ncols
# plt.figure(figsize=(ncols * 5, nrows * 5))
#
# for i, (dir, hist) in enumerate(mask_histograms.items()):
#     plt.subplot(nrows, ncols, i + 1)
#     plt.hist(hist, bins=100)
#     plt.title(f'Mask Histogram for {dir}')
#     plt.xlabel('Mask Bins')
#     plt.ylabel('Frequency')
#     print(dir, np.mean(hist))
#
# plt.tight_layout()
# plt.show()
#
# # Depth Histograms
# nrows = (len(depth_histograms) + ncols - 1) // ncols
# plt.figure(figsize=(ncols * 5, nrows * 5))
#
# for i, (dir, hist) in enumerate(depth_histograms.items()):
#     plt.subplot(nrows, ncols, i + 1)
#     plt.bar(range(100), hist, width=1.0, edgecolor='black')
#     plt.title(f'Depth Histogram for {dir}')
#     plt.xlabel('Depth Bins')
#     plt.ylabel('Frequency')
#
# plt.tight_layout()
# plt.show()