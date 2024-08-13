import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as pil
import time



def get_hist_parallel_with_torch_tracking(dep: torch.Tensor, simu_max_distance: float, zone_num: int) -> torch.Tensor:    # Share same interval/area

    height, width = dep.shape[1], dep.shape[2]


    range_margin = torch.linspace(0, simu_max_distance, int(simu_max_distance / 0.04) + 1,device=dep.device)

    patch_height, patch_width = height // zone_num, width // zone_num


    range_margin = torch.linspace(0, simu_max_distance, int(simu_max_distance / 0.04) + 1, device=dep.device)
    patch_height, patch_width = height // zone_num, width // zone_num

    offset = 0
    train_zone_num = zone_num

    sy = int((height - patch_height * train_zone_num) / 2) + offset
    sx = int((width - patch_width * train_zone_num) / 2) + offset
    dep_extracted = dep[:, sy:sy + patch_height * train_zone_num, sx:sx + patch_width * train_zone_num]
    # dep_patches = dep_extracted.unfold(2, patch_width, patch_width).unfold(1, patch_height, patch_height)
    # dep_patches = dep_patches.contiguous().view(-1, patch_height, patch_width)
    dep_patches = dep_extracted.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
    dep_patches = dep_patches.contiguous().view(-1, patch_height, patch_width)
    hist = torch.stack([torch.histc(x, bins=int(simu_max_distance / 0.04), min=0, max=simu_max_distance) for x in dep_patches], 0)

    # Initialize a tensor to track the pixel contributions
    pixel_contributions = torch.zeros_like(dep_extracted, dtype=torch.bool,device=dep.device)

    # Mask out depth smaller than 4cm, which is usually invalid depth, i.e., zero
    hist[:, 0] = 0
    hist = torch.clip(hist - 20, 0, None)



    for i, bin_data in enumerate(hist):
        idx = torch.nonzero(bin_data).squeeze(1)
        if idx.numel() == 0:
            continue

        # aa = time.time()
        diff_idx = torch.diff(idx)
        # split_indices = (diff_idx != 1).nonzero(as_tuple=False).squeeze(1) + 1
        split_indices = (diff_idx != 1).nonzero().squeeze(1) + 1
        # split_sizes = torch.diff(torch.cat((torch.tensor([0],device=dep.device), split_indices, torch.tensor([idx.numel()],device=dep.device)))).tolist()
        split_sizes_tensor = torch.diff(torch.cat((
            torch.tensor([0], device=dep.device),
            split_indices,
            torch.tensor([idx.numel()], device=dep.device)
        )))
        split_sizes = [int(size.item()) for size in split_sizes_tensor]
        # bb = time.time()
        idx_split = torch.split(idx, split_sizes)
        bin_data_split = [bin_data[ids] for ids in idx_split]
        bin_data_split_tensor = torch.stack([torch.sum(b) for b in bin_data_split])
        signal = torch.argmax(bin_data_split_tensor)
        # signal = torch.argmax(torch.tensor([torch.sum(b) for b in bin_data_split]))
        hist[i, :] = 0
        hist[i, idx_split[signal]] = bin_data_split[signal]
        # cc = time.time()
        # Determine which pixels contributed to this histogram
        patch_y = (i // train_zone_num) * patch_height
        patch_x = (i % train_zone_num) * patch_width
        patch = dep_patches[i]

        # dd = time.time()
        possible_values = [range_margin[idx_split[signal][0]], range_margin[idx_split[signal][-1]]+0.04]
        mask = (patch >= possible_values[0]) & (patch < possible_values[1])
        pixel_contributions[0, patch_y:patch_y + patch_height, patch_x:patch_x + patch_width] = mask
        # ee = time.time()
        # print(bb-aa,cc-bb,dd-cc,ee-dd)


    # dist = (range_margin[1:] + range_margin[:-1]) / 2
    # dist = dist.unsqueeze(0)
    # sy_indices = torch.arange(sy, sy + patch_height * train_zone_num, patch_height).repeat_interleave(train_zone_num)
    # sx_indices = torch.arange(sx, sx + patch_width * train_zone_num, patch_width).repeat(train_zone_num)
    # fr = torch.stack([sy_indices, sx_indices, sy_indices + patch_height, sx_indices + patch_width], dim=1)
    #
    # n = torch.sum(hist, dim=1)
    # mask = n > 0
    # mu = torch.sum(dist * hist, dim=1) / (n + 1e-9)
    # std = torch.sqrt(torch.sum(hist * torch.pow(dist - mu.unsqueeze(-1), 2), dim=1) / (n + 1e-9)) + 1e-9
    # fh = torch.stack([mu, std], dim=1).reshape([train_zone_num, train_zone_num, 2])
    # fh = fh.reshape([-1, 2])
    return pixel_contributions













depth_path = "/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/nyu2_test/00000_depth.png"
rgb_path = "/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/nyu2_test/00000_colors.png"

depth_gt = pil.open(depth_path)
depth_gt = np.array(depth_gt).astype(np.float32) / 1000
depth_gt = torch.Tensor(depth_gt).unsqueeze(0).cuda()
rgb = pil.open(rgb_path)
rgb = np.array(rgb).astype(np.float32) / 255
#
# plt.imshow(rgb)
# plt.show()
# plt.imshow(depth_gt[0].detach().cpu().numpy())
# plt.show()

opt = type('opt', (object,), {})()
opt.simu_max_distance = 10.0
opt.zone_num = 8
opt.oracle = True

times = []
for i in range(100):
    aa = time.time()
    map = get_hist_parallel_with_torch_tracking(depth_gt, opt.simu_max_distance, opt.zone_num)
    bb = time.time()
    times.append(bb-aa)
print(np.mean(times))




# print(map.shape)
# plt.imshow(map[0].detach().cpu().numpy())
# plt.show()

