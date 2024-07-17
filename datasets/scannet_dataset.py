# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import math
from einops import rearrange
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))


class ScannetTestPoseDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 is_train=False,
                 ):
        super(ScannetTestPoseDataset, self).__init__()
        self.full_res_shape = (1296, 968) 
        self.K = self._get_intrinsics()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
                (self.height, self.width),
                interpolation=self.interp
        )

        self.load_depth = False

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        line = self.filenames[index].split()
        line = [os.path.join(self.data_path, item) for item in line]

        for ind, i in enumerate(self.frame_idxs):
            inputs[("color", i, -1)] = self.get_color(line[ind])

        K = self.K.copy()
        this_width = self.width 
        this_height= self.height 

        K[0, :] *= this_width 
        K[1, :] *= this_height

        inv_K = np.linalg.pinv(K)

        inputs[("K")] = torch.from_numpy(K).float()
        inputs[("inv_K")] = torch.from_numpy(inv_K).float()
        
        # self.preprocess(inputs)

        for i in self.frame_idxs:
            inputs[('color', i, 0)] = self.to_tensor(
                    self.resize(inputs[('color', i,  -1)])
            )
            del inputs[("color", i, -1)]
   

        if self.load_depth:
            for ind, i in enumerate(self.frame_idxs):
                this_depth = line[ind].replace('color', 'depth').replace('.jpg', '.png')
                this_depth = cv2.imread(this_depth, -1) / 1000.0
                this_depth = cv2.resize(this_depth, (self.width, self.height))
                this_depth = self.to_tensor(this_depth)

                # assume no flippling
                inputs[("depth", i)] = this_depth

                if 'tof' in self.opt.sparse_d_type:
                    tof_depth, additional = get_tof_depth(self,inputs[("depth", i)])
                    if self.opt.sparse_d_type == 'tof_disp':
                        valid = tof_depth > 0
                        tof_depth = 1. / (tof_depth + 1e-7)
                        tof_depth[~valid] = 0
                    inputs[("tof_depth", i)] = tof_depth
                    inputs[("additional", i)] = additional


        pose1_dir = line[0].replace('color', 'pose').replace('.jpg', '.txt')
        pose2_dir = line[1].replace('color', 'pose').replace('.jpg', '.txt')
        pose1 = np.loadtxt(pose1_dir, delimiter=' ')
        pose2 = np.loadtxt(pose2_dir, delimiter=' ')
        pose_gt = np.dot(np.linalg.inv(pose2), pose1)
        inputs['pose_gt'] = pose_gt
        
        return inputs

    def get_color(self, fp):
        color = self.loader(fp)
        return Image.fromarray(color)

    def check_depth(self):
        return False

    def _get_intrinsics(self):
        w, h = self.full_res_shape
        intrinsics =np.array([[1161.75/w, 0., 658.042/w, 0.], 
                               [0., 1169.11/h, 486.467/h, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics


class ScannetTestDepthDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
        ):
        super(ScannetTestDepthDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
            (self.height, self.width),
            interpolation=self.interp

        )

    def __len__(self):
        return len(self.filenames)

    def _get_intrinsics(self):
        w, h = (1296, 968)
        intrinsics =np.array([[1161.75/w, 0., 658.042/w, 0.],
                               [0., 1169.11/h, 486.467/h, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics

    def __getitem__(self, index):

        color_name = self.filenames[index].replace('/', '_')
        color_path = os.path.join(self.data_path, color_name)
        depth_path = color_path.replace('color', 'depth').replace('jpg', 'png')
        
        rgb = self.loader(color_path)
        depth = cv2.imread(depth_path, -1) / 1000
       
        rgb = Image.fromarray(rgb)

        rgb = self.to_tensor(self.resize(rgb))
        depth = self.to_tensor(depth)

        K = self._get_intrinsics()
        K[0, :] *= self.width
        K[1, :] *= self.height

        Us, Vs = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.float32),
                             np.linspace(0, self.height - 1, self.height, dtype=np.float32), indexing='xy')
        Ones = np.ones([self.height, self.width], dtype=np.float32)
        norm_pix_coords = np.stack(((Us - K[0, 2]) / K[0, 0], (Vs - K[1, 2]) / K[1, 1], Ones), axis=0)
        norm_pix_coords = torch.from_numpy(norm_pix_coords)
        
        return rgb, depth, norm_pix_coords


def get_hist_parallel(self,dep):
    # share same interval/area
    height, width = dep.shape[1], dep.shape[2]

    max_distance = self.opt.simu_max_distance
    range_margin = list(np.arange(0, max_distance + 1e-9, 0.04))

    patch_height, patch_width = 32, 32
    offset = 0
    train_zone_num = self.opt.zone_num
    sy = int((height - patch_height * train_zone_num) / 2) + offset
    sx = int((width - patch_width * train_zone_num) / 2) + offset
    dep_extracted = dep[:, sy:sy + patch_height * train_zone_num, sx:sx + patch_width * train_zone_num]
    dep_patches = dep_extracted.unfold(2, patch_width, patch_width).unfold(1, patch_height, patch_height)
    dep_patches = dep_patches.contiguous().view(-1, patch_height, patch_width)
    hist = torch.stack(
        [torch.histc(x, bins=int(max_distance / 0.04), min=0, max=max_distance) for x in dep_patches], 0)

    # choose cluster with strongest signal, then fit the distribution
    # first, mask out depth smaller than 4cm, which is usually invalid depth, i.e., zero
    hist[:, 0] = 0
    hist = torch.clip(hist - 20, 0, None)
    for i, bin_data in enumerate(hist):
        idx = np.where(bin_data != 0)[0]
        idx_split = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        bin_data_split = np.split(bin_data[idx], np.where(np.diff(idx) != 1)[0] + 1)
        signal = np.argmax([torch.sum(b) for b in bin_data_split])
        hist[i, :] = 0
        hist[i, idx_split[signal]] = bin_data_split[signal]

    dist = ((torch.Tensor(range_margin[1:]) + np.array(range_margin[:-1])) / 2).unsqueeze(0)
    sy = torch.Tensor(list(range(sy, sy + patch_height * train_zone_num, patch_height)) * train_zone_num).view(
        [train_zone_num, -1]).T.reshape([-1])
    sx = torch.Tensor(list(range(sx, sx + patch_width * train_zone_num, patch_width)) * train_zone_num)
    fr = torch.stack([sy, sx, sy + patch_height, sx + patch_width], dim=1)

    mask = torch.zeros([train_zone_num, train_zone_num], dtype=torch.bool)
    n = torch.sum(hist, dim=1)
    mask = n > 0
    mask = mask.reshape([-1])
    mu = torch.sum(dist * hist, dim=1) / (n + 1e-9)
    std = torch.sqrt(torch.sum(hist * torch.pow(dist - mu.unsqueeze(-1), 2), dim=1) / (n + 1e-9)) + 1e-9
    fh = torch.stack([mu, std], axis=1).reshape([train_zone_num, train_zone_num, 2])
    fh = fh.reshape([-1, 2])

    return fh, fr, mask

def get_tof_depth(self, full_depth):
    full_depth = torch.nn.functional.interpolate(full_depth.unsqueeze(0), (self.opt.height, self.opt.width), mode='nearest')[0]
    hist_data, fr, mask = get_hist_parallel(self,full_depth)
    original_mask = mask.clone()
    original_hist_data = hist_data.clone()


    # fh = sample_point_from_hist_parallel(hist_data, mask, self.args)
    # patch_info = patch_info_from_rect_data(fr)
    my_mask = torch.zeros_like(full_depth)
    aa, bb = fr[0, :2].int()
    cc, dd = fr[-1, 2:].int()
    aa, bb = max(0, aa), max(0, bb)
    cc, dd = min(full_depth.shape[1], cc), min(full_depth.shape[2], dd)
    my_mask[:, aa:cc, bb:dd, ] = 1

    if self.opt.sparse_depth_input_type == 'area_mean':
        tof_depth = torch.zeros_like(full_depth)
        n = int(math.sqrt(hist_data.shape[0]))
        p1, p2 = torch.div(cc - aa, n, rounding_mode='floor'), torch.div(dd - bb, n, rounding_mode='floor')
        tof_unfold = (hist_data[:, 0] * mask).unsqueeze(0).unsqueeze(-1).repeat(1, 1, p1 * p2)
        tof_depth[:, aa:cc, bb:dd] = rearrange(tof_unfold, ' c (zn1 zn2) (p1 p2) ->c (zn1 p1) (zn2 p2)', zn1=n,
                                               zn2=n, p1=p1, p2=p2)
    elif 'area_multi' in self.opt.sparse_depth_input_type:
        tof_depth = torch.zeros_like(full_depth).repeat(int(self.opt.sparse_depth_input_type.split('_')[-1]), 1, 1)
        n = int(math.sqrt(hist_data.shape[0]))
        p1, p2 = torch.div(cc - aa, n, rounding_mode='floor'), torch.div(dd - bb, n, rounding_mode='floor')
        self.opt.zone_sample_num = int(self.opt.sparse_depth_input_type.split('_')[-1])
        fh = self.sample_point_from_hist_parallel(hist_data, mask, self.opt)
        tof_unfold = (fh * mask.unsqueeze(-1)).permute(1, 0).unsqueeze(-1).repeat(1, 1, p1 * p2)
        tof_depth[:, aa:cc, bb:dd] = rearrange(tof_unfold, ' c (zn1 zn2) (p1 p2) ->c (zn1 p1) (zn2 p2)', zn1=n,
                                               zn2=n, p1=p1, p2=p2)
    elif self.opt.sparse_depth_input_type == 'single_mean':
        tof_depth = torch.zeros_like(full_depth)
        center_x = torch.floor((fr[mask > 0][:, 0] + fr[mask > 0][:, 2]) / 2).long()
        center_y = torch.floor((fr[mask > 0][:, 1] + fr[mask > 0][:, 3]) / 2).long()
        tof_depth[:, center_x, center_y] = hist_data[mask, 0].unsqueeze(0).float()

    if 'area' in self.opt.sparse_depth_input_type and self.opt.extend_fov:
        fr_grid = fr.reshape(self.opt.zone_num, self.opt.zone_num, 4)
        mask_grid = mask.reshape(self.opt.zone_num, self.opt.zone_num)
        for i in [0, -1]:
            for j in range(1, self.opt.zone_num - 1):
                if mask_grid[i, j] == 0:
                    continue
                aaa, bbb, ccc, ddd = fr_grid[i, j].int()
                aaa, bbb = max(0, aaa), max(0, bbb)
                ccc, ddd = min(full_depth.shape[1], ccc), min(full_depth.shape[2], ddd)
                cropped_tof = tof_depth[:, aaa:ccc, bbb:ddd]
                if i == 0:
                    tof_depth[:, :aaa, bbb:ddd] = cropped_tof.mean([1, 2], True)
                else:
                    tof_depth[:, ccc:, bbb:ddd] = cropped_tof.mean([1, 2], True)
        for j in [0, -1]:
            for i in range(1, self.opt.zone_num - 1):
                if mask_grid[i, j] == 0:
                    continue
                aaa, bbb, ccc, ddd = fr_grid[i, j].int()
                aaa, bbb = max(0, aaa), max(0, bbb)
                ccc, ddd = min(full_depth.shape[1], ccc), min(full_depth.shape[2], ddd)
                cropped_tof = tof_depth[:, aaa:ccc, bbb:ddd]
                if j == 0:
                    tof_depth[:, aaa:ccc, :bbb] = cropped_tof.mean([1, 2], True)
                else:
                    tof_depth[:, aaa:ccc, ddd:] = cropped_tof.mean([1, 2], True)
        for i in [0, -1]:
            for j in [0, -1]:
                if mask_grid[i, j] == 0:
                    continue
                aaa, bbb, ccc, ddd = fr_grid[i, j].int()
                aaa, bbb = max(0, aaa), max(0, bbb)
                ccc, ddd = min(full_depth.shape[1], ccc), min(full_depth.shape[2], ddd)
                cropped_tof = tof_depth[:, aaa:ccc, bbb:ddd]
                if i == 0 and j == 0:
                    tof_depth[:, :ccc, :ddd] = cropped_tof.mean([1, 2], True)
                elif i == 0 and j == -1:
                    tof_depth[:, :ccc, bbb:] = cropped_tof.mean([1, 2], True)
                elif i == -1 and j == 0:
                    tof_depth[:, aaa:, :ddd] = cropped_tof.mean([1, 2], True)
                else:
                    tof_depth[:, aaa:, bbb:] = cropped_tof.mean([1, 2], True)
    additional = {
        'hist_data': hist_data.to(torch.float),
        'original_hist_data': original_hist_data.to(torch.float),
        'rect_data': fr.to(torch.float),
        'mask': mask,
        'original_mask': original_mask,
        # 'patch_info': patch_info,
        'my_mask': my_mask,
        'tof_depth': tof_depth
    }
    # plt.imshow(image.transpose(1,2,0)/255.)
    #
    # for i, xxx in enumerate(fr):
    #     a, b, c, d = xxx
    #     if mask[i]:
    #         plt.gca().add_patch(Rectangle((b, a), c - a, d - b,
    #                                   edgecolor='red',
    #                                   facecolor='none',
    #                                   lw=4))
    # plt.show()
    #
    #
    # plt.imshow(full_depth.permute(1,2,0))
    # plt.colorbar()
    # for i, xxx in enumerate(fr):
    #     a, b, c, d = xxx
    #     if mask[i]:
    #         plt.gca().add_patch(Rectangle((b, a), c - a, d - b,
    #                                   edgecolor='red',
    #                                   facecolor='none',
    #                                   lw=4))
    # plt.show()
    #
    # plt.imshow(image.transpose(1,2,0)/255.)
    #
    # for i, xxx in enumerate(fr):
    #     a, b, c, d = xxx
    #     if mask[i]:
    #         plt.gca().add_patch(Rectangle((b, a), c - a, d - b,
    #                                       edgecolor='red',
    #                                       facecolor='none',
    #                                       lw=4))
    # plt.show()
    #
    # plt.imshow(sample['depth'].permute(1, 2, 0))
    # plt.colorbar()
    # for i, xxx in enumerate(fr):
    #     a, b, c, d = xxx
    #     if mask[i]:
    #         plt.gca().add_patch(Rectangle((b, a), c - a, d - b,
    #                                       edgecolor='red',
    #                                       facecolor='none',
    #                                       lw=4))
    # plt.show()
    #
    # plt.imshow(hist_data[:, 0].reshape(8, 8))
    # plt.colorbar()
    # plt.show()
    # plt.imshow(hist_data[:, 1].reshape(8, 8))
    # plt.colorbar()
    # plt.show()
    # plt.imshow(fh.mean(1).reshape(8, 8))
    # plt.colorbar()
    # plt.show()

    return tof_depth, additional

