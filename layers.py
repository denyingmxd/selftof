# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import spconv.pytorch as spconv
# from MinkowskiEngine import MinkowskiConvolution, MinkowskiELU
from attn import *
def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's disparity output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    scaled_disp = 1e-7 + disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def coeff_to_normal(coeff):
    """Convert network's coefficient output into surface normal
    """
    normal = -F.normalize(coeff, p=2, dim=1)
    return normal


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, i=-1,type=-1):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1 , 1, 0)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Convkxk(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_refl=True):
        super(Convkxk, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(padding)
        else:
            self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock_Sub(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels,padding=0):
        super(ConvBlock_Sub, self).__init__()

        self.conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels, 3, 1, padding),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class ConvBlock_Sub1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_Sub1x1, self).__init__()

        self.conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels, 1, 1, 0),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
class ConvBlock_Sparse(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_Sparse, self).__init__()

        self.conv = spconv.SparseSequential(
            spconv.SparseConv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class ConvBlock_MK(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_MK, self).__init__()
        from MinkowskiEngine import MinkowskiConvolution, MinkowskiBatchNorm, MinkowskiReLU,MinkowskiELU
        from MinkowskiEngine.MinkowskiOps import to_sparse
        self.conv = nn.Sequential(
            MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=1, dimension=2),
            MinkowskiELU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out



class ConvBlock_2(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, i, type=0):
        super(ConvBlock_2, self).__init__()
        kernel_sizes = [3, 3, 3, 3, 1]
        paddings = [(k -1) // 2 for k in kernel_sizes]
        # self.conv = Convkxk(in_channels, out_channels, kernel_sizes[i], paddings[i])

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i],
                                   padding=paddings[i], padding_mode='reflect')

        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlock_3(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, i, type=0):
        super(ConvBlock_3, self).__init__()
        kernel_sizes = [3, 3, 3, 3, 3]
        paddings = [(k -1) // 2 for k in kernel_sizes]
        # self.conv = Convkxk(in_channels, out_channels, kernel_sizes[i], paddings[i])

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i],
                                   padding=paddings[i], padding_mode='reflect')

        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlock_4(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, i, type=0):
        super(ConvBlock_4, self).__init__()
        kernel_sizes = [11, 9, 7, 5, 3]
        paddings = [(k -1) // 2 for k in kernel_sizes]
        # self.conv = Convkxk(in_channels, out_channels, kernel_sizes[i], paddings[i])
        def get_largest_common_divisor(a, b):
            while b:
                a, b = b, a % b
            return a
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i],
                                   padding=paddings[i], padding_mode='reflect',
                              groups=get_largest_common_divisor(in_channels,out_channels))

        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock_5(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, i, type=0):
        super(ConvBlock_5, self).__init__()
        kernel_sizes = [31, 17, 9, 5, 3]
        paddings = [(k -1) // 2 for k in kernel_sizes]
        # self.conv = Convkxk(in_channels, out_channels, kernel_sizes[i], paddings[i])
        def get_largest_common_divisor(a, b):
            while b:
                a, b = b, a % b
            return a
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i],
                                   padding=paddings[i], padding_mode='reflect',
                              groups=get_largest_common_divisor(in_channels,out_channels))

        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlock_6(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, i, type=0):
        super(ConvBlock_6, self).__init__()
        kernel_sizes = [11, 9, 7, 5, 3]
        paddings = [(k -1) // 2 for k in kernel_sizes]
        # self.conv = Convkxk(in_channels, out_channels, kernel_sizes[i], paddings[i])

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[i],
                                   padding=paddings[i], padding_mode='reflect',
                              groups=in_channels)

        self.nonlin = nn.ELU(inplace=True)
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_sizes[i],
                                      padding=paddings[i], padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=1, padding_mode='reflect')
    def forward(self, x):
        out = self.conv(x)
        mask = self.mask_conv(x)
        mask = self.sigmoid(mask)
        out = x+out*mask
        out = self.conv2(out)
        out = self.nonlin(out)
        return out


class ConvBlock_7(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, i, type=0):
        super(ConvBlock_7, self).__init__()
        kernel_sizes = [11, 9, 7, 5, 3]
        paddings = [(k -1) // 2 for k in kernel_sizes]
        # self.conv = Convkxk(in_channels, out_channels, kernel_sizes[i], paddings[i])

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[i],
                                   padding=paddings[i], padding_mode='reflect',
                              groups=in_channels)

        self.nonlin = nn.ELU(inplace=True)
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=3,
                                      padding=1, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=1, padding_mode='reflect')
    def forward(self, x):
        out = self.conv(x)
        mask = self.mask_conv(x)
        mask = self.sigmoid(mask)
        out = x+out*mask
        out = self.conv2(out)
        out = self.nonlin(out)
        return out




class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

    def forward(self, depth, norm_pix_coords):
        cam_points = depth * norm_pix_coords
        cam_points = torch.cat([cam_points.view(self.batch_size, cam_points.shape[1], -1), self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords[:, 0, :, :] /= self.width - 1
        pix_coords[:, 1, :, :] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    default: delta 1, first order gradient
    """

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def enlarge_mask(mask, kernel_size=3):
    """Layer to enlarge binary mask
    """
    B,C,H,W = mask.shape
    kernel = torch.ones((B, C, kernel_size, kernel_size), dtype=torch.float32).to(device=mask.device)

    # Use convolution to find neighbors
    neighbors = F.conv2d(mask, kernel, padding=kernel_size//2)

    # Any original zero value adjacent to a non-zero value is set to 1
    pad_ones = (neighbors > 0) & (mask == 0)
    mask = mask[0, 0] + pad_ones.float()  # Add to the original tensor

    return mask


def get_zone_smooth_loss(disp, img, inputs,args):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    default: delta 1, first order gradient
    """

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    if inputs is not None:
        mean_tof = inputs[('additional',0)]['tof_mean']
        weight_x = (torch.abs(mean_tof[:, :, :, :-1] - mean_tof[:, :, :, 1:])>0).float()
        weight_y = (torch.abs(mean_tof[:, :, :-1, :] - mean_tof[:, :, 1:, :])>0).float()

        if args.dilate_zone_boundary_rate>0:
            assert args.dilate_zone_boundary_rate%2==1
            weight_x = enlarge_mask(weight_x, kernel_size=args.dilate_zone_boundary_rate)
            weight_y = enlarge_mask(weight_y, kernel_size=args.dilate_zone_boundary_rate)
            pass

        grad_disp_x = grad_disp_x * weight_x
        grad_disp_y = grad_disp_y * weight_y

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_plane_loss(plane_keysets, points_3d):

    bs, ch, _ = points_3d.shape

    start_points = torch.gather(points_3d, 2, torch.stack(ch * [plane_keysets[:, 0]], 1))
    end_points_A = torch.gather(points_3d, 2, torch.stack(ch * [plane_keysets[:, 1]], 1))
    end_points_B = torch.gather(points_3d, 2, torch.stack(ch * [plane_keysets[:, 2]], 1))
    end_points_C = torch.gather(points_3d, 2, torch.stack(ch * [plane_keysets[:, 3]], 1))

    vector_A = end_points_A - start_points
    vector_B = end_points_B - start_points
    vector_C = end_points_C - start_points

    AxB = torch.cross(vector_A, vector_B, dim=1)

    AxB_dot_C = torch.sum(AxB*vector_C, dim=1)

    plane_loss = torch.abs(AxB_dot_C).mean()

    return plane_loss


def get_line_loss(line_keysets, points_3d):
    bs, ch, _ = points_3d.shape

    start_points = torch.gather(points_3d, 2, torch.stack(ch * [line_keysets[:, 0]], 1))
    end_points_A = torch.gather(points_3d, 2, torch.stack(ch * [line_keysets[:, 1]], 1))
    end_points_B = torch.gather(points_3d, 2, torch.stack(ch * [line_keysets[:, 2]], 1))

    vector_A = end_points_A - start_points
    vector_B = end_points_B - start_points

    AxB = torch.cross(vector_A, vector_B, dim=1)

    line_loss = torch.norm(AxB, p=2, dim=1).mean()

    return line_loss

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    log10 = torch.mean(torch.abs(torch.log10(pred / gt)))

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


def get_weighted_mean_std(folded_output_depth0,org_mask):
    assert  len(folded_output_depth0.shape)==4
    assert  len(org_mask.shape)==4
    assert folded_output_depth0.shape == org_mask.shape

    mask = org_mask.float().clone()

    sum_mask = mask.sum(-1)+1e-7
    valid_mask = (sum_mask >= 1e-5).float()
    # sum_mask = sum_mask + (sum_mask<1e-3 ).float()*1e-3
    # sum_mask = torch.ones_like(mask).sum(-1)
    means = ((folded_output_depth0 * mask).sum(-1) / sum_mask).squeeze(1)
    extended_means = means.unsqueeze(1).unsqueeze(-1)
    stds = ((((folded_output_depth0 - extended_means) ** 2)*mask).sum(-1) / (sum_mask)+1e-7).sqrt().squeeze(1)

    # log_vars = torch.log((((folded_output_depth0 - extended_means) ** 2) * mask).sum(-1) / sum_mask + 1e-7).squeeze(1)
    # stds = torch.exp(log_vars / 2)

    return means, stds, valid_mask.squeeze(1)


def compute_sparse_depth_loss(sparse_depth0, output_depth0,inputs,outputs,args):
    additional = inputs[('additional',0)]
    if 'area' in args.sparse_depth_loss_type and 'tof' in args.sparse_d_type:
        B,C,H,W = output_depth0.shape
        n = int(math.sqrt(additional['hist_data'].to(output_depth0.device).shape[1]))
        my_mask = additional['my_mask'].to(output_depth0.device)

        hist_data = additional['hist_data'].to(output_depth0.device)
        hist_mask = additional['mask'].to(output_depth0.device)

        rect_data = additional['rect_data'].to(output_depth0.device).long()
        tof_mask = additional['tof_mask'].to(output_depth0.device).long()



        aa, bb, cc, dd = rect_data[:, 0, 0][0], rect_data[:, 0, 1][0], rect_data[:, -1, 2][0], rect_data[:, -1, 3][0]
        p1, p2 = torch.div(cc - aa, n, rounding_mode='floor'), torch.div(dd - bb, n, rounding_mode='floor')
        cropped_output_depth0 = output_depth0[:, :, aa:cc, bb:dd]
        folded_output_depth0 = rearrange(cropped_output_depth0, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)',zn1=n, zn2=n, p1=p1, p2=p2).contiguous()
        if args.load_seg:
            pred_seg = inputs[('seg', 0, 0)]
            cropped_pred_seg = pred_seg[:, :, aa:cc, bb:dd]
            folded_pred_seg = rearrange(cropped_pred_seg, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)',zn1=n, zn2=n, p1=p1, p2=p2).contiguous()
        #maybe with rearrange is faster, stilkl think about bincount maybe slow

            flatten_folded_pred_seg = rearrange(folded_pred_seg, 'b 1 (zn1 zn2) (p1 p2) -> (b zn1 zn2) (p1 p2)',zn1=n, zn2=n, p1=p1, p2=p2)
            # seg_counts = [torch.bincount(flatten_folded_pred_seg[i]) for i in range(len(flatten_folded_pred_seg))]
            # seg_counts = [torch.bincount(flatten_folded_pred_seg[i],minlength=150) for i in range(len(flatten_folded_pred_seg))]
            seg_counts = [torch.histc(flatten_folded_pred_seg[i], bins=150, min=0, max=149) for i in range(len(flatten_folded_pred_seg))]
            argmax_seg = [torch.argmax(seg_count) for seg_count in seg_counts]
            argmax_seg_mask = [flatten_folded_pred_seg[i]==argmax_seg[i] for i in range(len(flatten_folded_pred_seg))]
            argmax_seg_mask = torch.stack(argmax_seg_mask,0)

            flatten_argmax_seg_mask = rearrange(argmax_seg_mask, '(b zn1 zn2) (p1 p2) -> b 1 (zn1 zn2) (p1 p2)',zn1=n, zn2=n, p1=p1, p2=p2)
            valid_area_tensor = rearrange(flatten_argmax_seg_mask, 'b 1 (zn1 zn2) (p1 p2) -> b 1 (zn1 p1) (zn2 p2)',zn1=n, zn2=n, p1=p1, p2=p2)

            flatten_argmax_seg_mask = flatten_argmax_seg_mask.float()
            # flatten_argmax_seg_mask[flatten_argmax_seg_mask==0]=0.1
            means,stds,valid_mask = get_weighted_mean_std(folded_output_depth0,flatten_argmax_seg_mask)



        else:
            valid_area_tensor = torch.ones((B,1,cc-aa,dd-bb),device=sparse_depth0.device)
            if args.weighted_ls_type>0:
                if args.weighted_ls_type==1:
                    distance = ((folded_output_depth0-hist_data[:,:,0].unsqueeze(1).unsqueeze(-1))**2)/2
                    # distance = torch.clamp(distance,1e-3,10)
                    weights_by_distance = torch.exp(-distance)

                elif args.weighted_ls_type==3:
                    distance = ((folded_output_depth0-folded_output_depth0.mean([3]).unsqueeze(-1))**2)/2
                    # distance = torch.clamp(distance,1e-3,10)
                    weights_by_distance = torch.exp(-distance)
                elif args.weighted_ls_type==4:
                    distance = ((folded_output_depth0 - torch.median(folded_output_depth0, dim=-1, keepdim=True).values) ** 2) / 2
                    weights_by_distance = torch.exp(-distance)
                elif args.weighted_ls_type==6:
                    distance = ((folded_output_depth0 - torch.median(folded_output_depth0, dim=-1, keepdim=True).values) ** 2) / 1
                    weights_by_distance = torch.exp(-distance)
                elif args.weighted_ls_type==7:
                    distance = ((folded_output_depth0 - torch.median(folded_output_depth0, dim=-1, keepdim=True).values) ** 2) / 5
                    weights_by_distance = torch.exp(-distance)
                elif args.weighted_ls_type==8:
                    distance = ((folded_output_depth0 - torch.median(folded_output_depth0, dim=-1, keepdim=True).values) ** 2) / 10
                    weights_by_distance = torch.exp(-distance)
                elif args.weighted_ls_type==9:
                    distance = ((folded_output_depth0 - torch.median(folded_output_depth0, dim=-1, keepdim=True).values) ** 2) / 20
                    weights_by_distance = torch.exp(-distance)
                elif args.weighted_ls_type==10:
                    distance = ((folded_output_depth0 - torch.median(folded_output_depth0, dim=-1, keepdim=True).values) ** 2) / 0.5
                    weights_by_distance = torch.exp(-distance)
                elif args.weighted_ls_type==11:
                    distance = ((folded_output_depth0 - torch.median(folded_output_depth0, dim=-1, keepdim=True).values) ** 2) / 0.25
                    weights_by_distance = torch.exp(-distance)
                elif args.weighted_ls_type==5:
                    distance = ((folded_output_depth0 - torch.median(folded_output_depth0, dim=-1, keepdim=True).values) ** 2) / 2
                    weights_by_distance = torch.exp(-distance)
                    weights_by_distance = weights_by_distance.float()
                    means, stds, valid_mask = get_weighted_mean_std(folded_output_depth0, weights_by_distance)
                    distance = torch.abs(folded_output_depth0 - means.unsqueeze(1).unsqueeze(-1))
                    weights_by_distance = distance <= (3 * stds.unsqueeze(1).unsqueeze(-1))
                elif args.weighted_ls_type==12:
                    if ('pixel_distributions',0) in outputs.keys():
                        weights_by_distance = outputs[('pixel_distributions',0)]
                    else:
                        weights_by_distance = [get_hist_parallel_with_torch_tracking(output_depth0[i],args) for i in range(output_depth0.shape[0])]
                        weights_by_distance = torch.stack(weights_by_distance,0)
                    # weights_by_distance = get_hist_parallel_with_torch_tracking_batch(output_depth0)
                    weights_by_distance = rearrange(weights_by_distance, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)', zn1=n, zn2=n,
                              p1=p1, p2=p2).contiguous()


                elif args.weighted_ls_type==13:
                    pred_seg = inputs[('seg', 0, 0)]
                    cropped_pred_seg = pred_seg[:, :, aa:cc, bb:dd]
                    folded_pred_seg = rearrange(cropped_pred_seg, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)',
                                                zn1=n, zn2=n, p1=p1, p2=p2).contiguous()
                    flatten_folded_pred_seg = rearrange(folded_pred_seg, 'b 1 (zn1 zn2) (p1 p2) -> (b zn1 zn2) (p1 p2)',
                                                        zn1=n, zn2=n, p1=p1, p2=p2)
                    seg_counts = [torch.histc(flatten_folded_pred_seg[i], bins=150, min=0, max=149) for i in
                                  range(len(flatten_folded_pred_seg))]
                    argmax_seg = [torch.argmax(seg_count) for seg_count in seg_counts]
                    argmax_seg_mask = [flatten_folded_pred_seg[i] == argmax_seg[i] for i in
                                       range(len(flatten_folded_pred_seg))]
                    argmax_seg_mask = torch.stack(argmax_seg_mask, 0)

                    flatten_argmax_seg_mask = rearrange(argmax_seg_mask, '(b zn1 zn2) (p1 p2) -> b 1 (zn1 zn2) (p1 p2)',
                                                        zn1=n, zn2=n, p1=p1, p2=p2)
                    valid_area_tensor = rearrange(flatten_argmax_seg_mask,
                                                  'b 1 (zn1 zn2) (p1 p2) -> b 1 (zn1 p1) (zn2 p2)', zn1=n, zn2=n, p1=p1,
                                                  p2=p2)
                    weights_by_distance = [get_hist_parallel_with_torch_tracking(cropped_output_depth0[i],args) for i in
                                           range(cropped_output_depth0.shape[0])]
                    weights_by_distance = torch.stack(weights_by_distance, 0)
                    # weights_by_distance = get_hist_parallel_with_torch_tracking_batch(output_depth0)

                    weights_by_distance = weights_by_distance * valid_area_tensor
                    weights_by_distance = rearrange(weights_by_distance,
                                                    'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)', zn1=n, zn2=n,
                                                    p1=p1, p2=p2).contiguous()
                    semantic_mask = valid_area_tensor




                weights_by_distance = weights_by_distance.float()
                weights_by_distance = weights_by_distance * hist_mask.unsqueeze(1).unsqueeze(-1).float()
                # weights_by_distance[weights_by_distance<=0.1]=0.1
                means,stds,valid_mask = get_weighted_mean_std(folded_output_depth0,weights_by_distance)
                weights_by_distance = weights_by_distance*(valid_mask.unsqueeze(1).unsqueeze(-1))

                outputs[('ls_weight_map', 0)] = rearrange(weights_by_distance, 'b 1 (zn1 zn2) (p1 p2) -> b 1 (zn1 p1) (zn2 p2)', zn1=n, zn2=n, p1=p1, p2=p2)


            else:
                valid_mask = torch.ones_like(hist_mask)
                means = folded_output_depth0.mean([3]).squeeze(1)
                stds = folded_output_depth0.std([3]).squeeze(1)

        # if args.sparse_depth_loss_type=='area_L1':
        #     mean_loss = torch.abs(mean - hist_data[:,:,0]) * hist_mask
        #     std_loss = torch.abs(std - hist_data[:,:,1]) * hist_mask
        # weights_gaussian_mle = torch.ones_like(folded_output_depth0)

        # weighted_means = (folded_output_depth0 * weights_gaussian_mle).sum([3,4]).squeeze(1) / weights_gaussian_mle.sum([3,4]).squeeze(1)


        if args.sparse_depth_loss_type=='area_L2':
            mean_loss = torch.square(means - hist_data[:, :, 0]) * hist_mask
            std_loss = torch.square(stds - hist_data[:, :, 1]) * hist_mask

        # if args.margin_level>0:
        #     margin_mask = mean_loss > hist_data[:,:,1]**2*args.margin_level
        #     mean_loss = mean_loss * margin_mask
        #     std_loss =  std_loss * margin_mask
        #     # hist_mask = hist_mask * margin_mask
        #     for b in range(B):
        #         for i in range(n**2):
        #             if margin_mask[b,i] and hist_mask[b,i]:
        #                 aa, bb, cc, dd = rect_data[b,i].long()
        #                 valid_area_tensor[b, :, aa:cc,bb:dd] = margin_mask[b,i].repeat(1,1,cc-aa,dd-bb)
        #
        # pass

        weight_map = torch.ones(B,n**2)
        if args.guided_loss_type==1:
            min_photo_loss =  torch.min(outputs[('reprojection_losses',2,0)],outputs[('reprojection_losses',-2,0)])
            cropped_min_photo_loss = min_photo_loss[:, :, aa:cc, bb:dd]
            folded_photo_loss = rearrange(cropped_min_photo_loss, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) p1 p2', zn1=n, zn2=n, p1=p1, p2=p2)
            folded_photo_loss_mean = folded_photo_loss.mean([3,4]).squeeze(1)
            mean_loss = mean_loss * torch.exp(1-folded_photo_loss_mean)
            std_loss = std_loss * torch.exp(1-folded_photo_loss_mean)
            weight_map = torch.exp(1-folded_photo_loss_mean)

        if args.guided_loss_type==2:
            with torch.no_grad():
                cropped_min_photo_loss = min_photo_loss[:, :, aa:cc, bb:dd]
                folded_photo_loss = rearrange(cropped_min_photo_loss, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) p1 p2',
                                              zn1=n, zn2=n, p1=p1, p2=p2)
                folded_photo_loss_mean = folded_photo_loss.mean([3,4]).squeeze(1).detach().clone()
            mean_loss = mean_loss * torch.exp(1-folded_photo_loss_mean)
            std_loss = std_loss * torch.exp(1-folded_photo_loss_mean)

        if args.guided_loss_type==3:
            folded_std = stds
            mean_loss = mean_loss * (torch.exp(-folded_std))
            std_loss = std_loss * (torch.exp(-folded_std))
            weight_map = torch.exp(-folded_std)

        if args.guided_loss_type==4:
            folded_std = stds
            mean_loss = mean_loss * (torch.exp(-0.5*folded_std))
            std_loss = std_loss * (torch.exp(-0.5*folded_std))
            weight_map = (torch.exp(-0.5*folded_std))

        if args.guided_loss_type==7:
            folded_std = stds
            mean_loss = mean_loss * (torch.exp(-2*folded_std))
            std_loss = std_loss * (torch.exp(-2*folded_std))
            weight_map = (torch.exp(-2*folded_std))


        if args.guided_loss_type==5:
            min_photo_loss =  torch.min(outputs[('reprojection_losses',2,0)],outputs[('reprojection_losses',-2,0)])
            cropped_min_photo_loss = min_photo_loss[:, :, aa:cc, bb:dd]
            folded_photo_loss = rearrange(cropped_min_photo_loss, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) p1 p2', zn1=n, zn2=n, p1=p1, p2=p2)
            folded_photo_loss_mean = folded_photo_loss.amin(dim=[3,4]).squeeze(1)
            mean_loss = mean_loss * torch.exp(1-folded_photo_loss_mean)
            std_loss = std_loss * torch.exp(1-folded_photo_loss_mean)
            weight_map = torch.exp(1-folded_photo_loss_mean)

        if args.guided_loss_type==6:
            min_photo_loss =  torch.min(outputs[('reprojection_losses',2,0)],outputs[('reprojection_losses',-2,0)])
            cropped_min_photo_loss = min_photo_loss[:, :, aa:cc, bb:dd]
            folded_photo_loss = rearrange(cropped_min_photo_loss, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) p1 p2', zn1=n, zn2=n, p1=p1, p2=p2)
            folded_photo_loss_mean = folded_photo_loss.amax(dim=[3,4]).squeeze(1)
            mean_loss = mean_loss * torch.exp(1-folded_photo_loss_mean)
            std_loss = std_loss * torch.exp(1-folded_photo_loss_mean)
            weight_map = torch.exp(1-folded_photo_loss_mean)


        # mean_loss = (mean_loss.sum(dim=1) / hist_mask.sum(dim=1)).mean()
        # std_loss = (std_loss.sum(dim=1) / hist_mask.sum(dim=1)).mean()
        mean_loss = ((mean_loss*valid_mask*hist_mask).sum(dim=1) / (valid_mask*hist_mask).sum(dim=1)).mean()
        std_loss = ((std_loss*valid_mask*hist_mask).sum(dim=1) / (valid_mask*hist_mask).sum(dim=1)).mean()
        outputs[('valid_con_mask',0)] = (valid_mask*hist_mask).reshape(B,1,n,n)
        # std_loss = torch.tensor(0.0).to(mean_loss.device)
        if torch.isnan(mean_loss).any() or torch.isnan(std_loss).any():
            print('nan')
            mean_loss = torch.tensor(0.0).to(mean_loss.device)
            std_loss = torch.tensor(0.0).to(std_loss.device)
            exit()
        outputs[('valid_area', 0)] = torch.zeros_like(sparse_depth0)
        outputs[('valid_area', 0)][:, :, aa:cc, bb:dd] = valid_area_tensor

        outputs[('con_weight_map', 0)] = rearrange(weight_map, 'b (zn1 zn2) -> b 1 (zn1) (zn2)', zn1=n, zn2=n)

        loss_sparse_depth = mean_loss + std_loss*args.std_weight
        
        if args.sl_type>0:
            # assert not torch.isnan(semantic_mask).any()
            if args.sl_type==1:
                distance = (folded_output_depth0 - means.unsqueeze(1).unsqueeze(-1))**2 / 2
                gaussian_mask = torch.exp(-distance)
                gaussian_mask = rearrange(gaussian_mask, 'b 1 (zn1 zn2) (p1 p2) -> b 1 (zn1 p1) (zn2 p2)', zn1=n, zn2=n,p1=p1, p2=p2)
                criterion = torch.nn.BCELoss(reduction='none')
                # Compute the binary cross-entropy loss
                sl_loss_full = criterion(gaussian_mask, semantic_mask.to(torch.float32))
                sl_loss = sl_loss_full.mean()
                # non_valid_mask = (~semantic_mask)
                # sl_loss = ((sl_loss * non_valid_mask).sum([2, 3]) / (non_valid_mask.sum([2, 3]) + 1e-7)).mean()


            outputs[('gaussian_mask', 0)] = gaussian_mask
            outputs[('sl_loss_full', 0)] = sl_loss_full
        else:
            sl_loss = torch.tensor(0.0).to(mean_loss.device)
            # outputs[('gaussian_mask', 0)] = torch.zeros_like(cropped_output_depth0)

    else:
        mean_loss, std_loss, loss_sparse_depth, sl_loss = torch.tensor(0.0).to(output_depth0.device), \
            torch.tensor(0.0).to(output_depth0.device), torch.tensor(0.0).to(output_depth0.device) \
            , torch.tensor(0.0).to(output_depth0.device)



    return mean_loss, std_loss, loss_sparse_depth, sl_loss

def scale_zone_depth_np(pred_depth,hist_data,p1,p2,n1,n2,n,pp=1,case='median_mul'):
    H,W = pred_depth.shape
    p1, p2 = p1*pp, p2*pp
    n1, n2 = n1//pp, n2//pp
    folded_output_depth0 = rearrange(pred_depth, '(zn1 p1) (zn2 p2) -> (zn1 zn2) p1 p2', zn1=n1, zn2=n2, p1=p1, p2=p2)
    pred_median = np.median(folded_output_depth0, axis=(1, 2))
    folded_hist = rearrange(hist_data[:,0].reshape(n,n), '(zn1 pp1) (zn2 pp2) -> (zn1 zn2) pp1 pp2', zn1=n1, zn2=n2, pp1=pp, pp2=pp)
    hist_median = np.median(folded_hist, axis=(1, 2))
    if case=='median_mul':
        scale_factor = hist_median / pred_median
        folded_output_depth0 *= scale_factor[:, None, None]
    elif case=='median_add':
        scale_factor = hist_median - pred_median
        folded_output_depth0 += scale_factor[:, None, None]
    elif case=='median_median_mul':
        scale_factor = hist_median / pred_median
        scale_factor = np.median(scale_factor)
        folded_output_depth0 *= scale_factor
    elif case=='median_mean_mul':
        scale_factor = hist_median / pred_median
        scale_factor = np.mean(scale_factor)
        folded_output_depth0 *= scale_factor
    folded_output_depth0 = rearrange(folded_output_depth0, '(zn1 zn2) p1 p2 -> (zn1 p1) (zn2 p2)', zn1=n1, zn2=n2, p1=p1, p2=p2)

    return folded_output_depth0

def get_4_neighbors(input):
    """
    Extract 4-neighbors (North, South, East, West) for each element in a BCHW tensor.

    Args:
    - input (torch.Tensor): The BCHW tensor.

    Returns:
    - torch.Tensor: Tensor containing the 4-neighbors for each element, with shape [B, C, H, W, 4].
    """
    # Padding the input tensor to handle edge cases
    padded_input = F.pad(input, (1, 1, 1, 1), mode='replicate', value=0)

    # Define kernels to extract each neighbor for all channels simultaneously
    # Create a single channel kernel and expand it to match the input channels
    north_kernel = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(input.device)
    south_kernel = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(input.device)
    east_kernel = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(input.device)
    west_kernel = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(input.device)

    # Expand kernels to the number of input channels
    num_channels = input.shape[1]
    north_kernel = north_kernel.repeat(1, num_channels, 1, 1)
    south_kernel = south_kernel.repeat(1, num_channels, 1, 1)
    east_kernel = east_kernel.repeat(1, num_channels, 1, 1)
    west_kernel = west_kernel.repeat(1, num_channels, 1, 1)

    # Apply convolution to extract each neighbor
    north = F.conv2d(padded_input, north_kernel, groups=num_channels)
    south = F.conv2d(padded_input, south_kernel, groups=num_channels)
    east = F.conv2d(padded_input, east_kernel, groups=num_channels)
    west = F.conv2d(padded_input, west_kernel, groups=num_channels)

    # Concatenate results to form a tensor of shape [B, C, H, W, 4]
    neighbors = torch.stack((north, south, east, west), dim=-1)

    return neighbors

def get_8_neighbors(input):
    """
    Extract 8-neighbors for each element in a BCHW tensor using unfold.

    Args:
    - input (torch.Tensor): Input tensor of shape [B, C, H, W].

    Returns:
    - torch.Tensor: Tensor containing the 8-neighbors for each element,
                    with shape [B, C, H, W, 8].
    """
    # Assuming the input tensor has padding of 1 pixel around it with 'constant' padding mode
    padded_input = torch.nn.functional.pad(input, (1, 1, 1, 1), mode='replicate', value=0)

    # Unfold the padded input to extract 3x3 patches
    # Resulting shape: [B, C*3*3, L], where L is the number of such blocks
    unfolded = torch.nn.functional.unfold(padded_input, kernel_size=3, stride=1, padding=0)

    # Reshape to separate channels and patches
    # New shape: [B, C, 3*3, H, W]
    batch_size, channels, height, width = input.shape
    unfolded = unfolded.view(batch_size, channels, 9, height, width)

    # Remove the center pixel and rearrange the tensor
    # Keeping only the neighbors and excluding the center (index 4)
    neighbors = torch.cat((unfolded[:, :, :4], unfolded[:, :, 5:]), dim=2)

    # Reshape to final desired shape: [B, C, H, W, 8]
    neighbors = neighbors.permute(0, 1, 3, 4, 2).contiguous()

    return neighbors


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, features,depth):

        h, w = depth.size()[2:]
        size = (h, w)
        low_feature = self.down_l(features)
        low_feature = F.upsample(low_feature, size, mode='bilinear', align_corners=True)#maybe nearest?
        flow = self.flow_make(low_feature)
        h_feature = self.flow_warp(depth, flow, size=size)

        return flow,h_feature


    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class AlignedModule2(nn.Module):

    def __init__(self, inplane, xplane, kernel_size=3):
        super(AlignedModule2, self).__init__()
        self.down_l = nn.Conv2d(inplane, inplane//2, 1, bias=False)
        self.down_h = nn.Conv2d(xplane, xplane//2, 1, bias=False)
        self.flow_make = nn.Conv2d((inplane//2+xplane//2), 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, features1, features2, depth):

        h, w = depth.size()[2:]
        size = (h, w)
        features1 = self.down_l(features1)
        features2 = self.down_h(features2)
        features1 = F.upsample(features1, size, mode='bilinear', align_corners=True)
        # features2 = F.upsample(features2, size, mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat([features1,features2],dim=1))
        h_feature = self.flow_warp(depth, flow, size=size)

        return flow,h_feature


    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output




def flow_to_image(flow, display=False):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def get_hist_parallel_with_torch_tracking(dep,args):
    # Share same interval/area

    height, width = dep.shape[1], dep.shape[2]

    max_distance = args.simu_max_distance
    range_margin = torch.linspace(0, max_distance, int(max_distance / 0.04) + 1).to(dep.device)

    patch_height, patch_width = height // args.zone_num, width // args.zone_num

    if args.oracle:
        args.simu_max_distance = 100.0
        max_distance = 100.0
        range_margin = torch.linspace(0, max_distance, int(max_distance / 0.04) + 1).to(dep.device)
        patch_height, patch_width = height // args.zone_num, width // args.zone_num

    offset = 0
    train_zone_num = args.zone_num

    sy = int((height - patch_height * train_zone_num) / 2) + offset
    sx = int((width - patch_width * train_zone_num) / 2) + offset
    dep_extracted = dep[:, sy:sy + patch_height * train_zone_num, sx:sx + patch_width * train_zone_num]
    # dep_patches = dep_extracted.unfold(2, patch_width, patch_width).unfold(1, patch_height, patch_height)
    # dep_patches = dep_patches.contiguous().view(-1, patch_height, patch_width)
    dep_patches = dep_extracted.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
    dep_patches = dep_patches.contiguous().view(-1, patch_height, patch_width)
    hist = torch.stack([torch.histc(x, bins=int(max_distance / 0.04), min=0, max=max_distance) for x in dep_patches], 0)

    # Initialize a tensor to track the pixel contributions
    pixel_contributions = torch.zeros_like(dep_extracted, dtype=torch.bool,device=dep.device)

    # Mask out depth smaller than 4cm, which is usually invalid depth, i.e., zero
    hist[:, 0] = 0
    hist = torch.clip(hist - 20, 0, None)

    for i, bin_data in enumerate(hist):
        idx = torch.nonzero(bin_data).squeeze(1)
        if idx.numel() == 0:
            continue
        diff_idx = torch.diff(idx)
        split_indices = (diff_idx != 1).nonzero(as_tuple=False).squeeze(1) + 1
        split_sizes = torch.diff(torch.cat((torch.tensor([0],device=dep.device), split_indices, torch.tensor([idx.numel()],device=dep.device)))).tolist()

        idx_split = torch.split(idx, split_sizes)
        bin_data_split = [bin_data[ids] for ids in idx_split]
        signal = torch.argmax(torch.tensor([torch.sum(b) for b in bin_data_split]))
        hist[i, :] = 0
        hist[i, idx_split[signal]] = bin_data_split[signal]

        # Determine which pixels contributed to this histogram
        patch_y = (i // train_zone_num) * patch_height
        patch_x = (i % train_zone_num) * patch_width
        patch = dep_patches[i]


        possible_values = [range_margin[idx_split[signal][0]], range_margin[idx_split[signal][-1]]+0.04]
        mask = (patch >= possible_values[0]) & (patch < possible_values[1])
        pixel_contributions[0, patch_y:patch_y + patch_height, patch_x:patch_x + patch_width] = mask
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






class AdditionNetwork_1(nn.Module):
    def __init__(self, num_ch_enc, args):
        super(AdditionNetwork_1, self).__init__()
        self.num_ch_enc = num_ch_enc
        #64,144,192
        #64, 72,96
        #128, 36, 48
        #256, 18, 24
        #512, 9, 12
        self.args = args
        self.convs = {}

        for i in range(len(num_ch_enc)-1,-1,-1):
        #     self.positional_encodings[i] = nn.Parameter(torch.rand(args.zone_num, num_ch_enc[i]), requires_grad=True)
        #     self.positional_encodings2[i] = nn.Parameter(torch.rand(args.zone_num, num_ch_enc[i]), requires_grad=True)
        # # https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L161
        #     nn.init.trunc_normal_(self.positional_encodings[i], std=0.2)
        #     nn.init.trunc_normal_(self.positional_encodings2[i], std=0.2)

            # self.cas[i] = LoFTREncoderLayer(num_ch_enc[i], 8)
            self.convs[('down',i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)


        self.addition = nn.ModuleList(list(self.convs.values()))
    def forward(self, rgb_features, depth_features,inputs):

        final_features = []
        for i in range(len(rgb_features)):
            rgb_feature = rgb_features[i]
            depth_feature = depth_features[i]

            rgb_feature_down = self.convs[('down',i)](rgb_feature)
            depth_features_low = torch.nn.functional.interpolate(depth_feature, size=(rgb_feature_down.shape[2], rgb_feature_down.shape[3]), mode='nearest')
            low_tof_mask = inputs[('additional', 0)]['mask'].reshape(-1,self.args.zone_num,self.args.zone_num)#4,64
            resized_tof_mask = torch.nn.functional.interpolate(low_tof_mask.unsqueeze(1).float(), size=(rgb_feature_down.shape[2], rgb_feature_down.shape[3]), mode='nearest')
            resized_tof_mask_flatten = resized_tof_mask.reshape(-1,rgb_feature_down.shape[2]*rgb_feature_down.shape[3])
            rgb_feature_down_flatten = rearrange(rgb_feature_down, 'b c h w -> b (h w) c')
            depth_feature_low_flatten = rearrange(depth_features_low, 'b c h w -> b (h w) c')
            rgb_affine = torch.matmul(rgb_feature_down_flatten, rgb_feature_down_flatten.transpose(1, 2))
            rgb_affine[resized_tof_mask_flatten.unsqueeze(1).repeat(1,rgb_affine.shape[-1],1) == 0.] = -1e9
            rgb_affine = torch.nn.functional.softmax(rgb_affine, dim=2)
            depth_new_flatten = torch.matmul(rgb_affine, depth_feature_low_flatten)
            depth_new = rearrange(depth_new_flatten, 'b (h w) c -> b c h w', h=rgb_feature_down.shape[2], w=rgb_feature_down.shape[3])
            depth_new = torch.nn.functional.interpolate(depth_new, size=(rgb_feature.shape[2], rgb_feature.shape[3]), mode='nearest')
            final_features.append(depth_new+rgb_feature)

        return final_features

class AdditionNetwork_2(nn.Module):
    def __init__(self, num_ch_enc, args):
        super(AdditionNetwork_2, self).__init__()
        self.num_ch_enc = num_ch_enc
        #64,144,192
        #64, 72,96
        #128, 36, 48
        #256, 18, 24
        #512, 9, 12
        self.args = args
        self.convs = {}
        self.position_embeddings = {}

        for i in range(len(num_ch_enc)-1,-1,-1):
            self.position_embeddings[str(i)] = nn.Parameter(torch.rand(9*12, num_ch_enc[i]), requires_grad=True)
        #     self.positional_encodings2[i] = nn.Parameter(torch.rand(args.zone_num, num_ch_enc[i]), requires_grad=True)
        # # https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L161
        #     nn.init.trunc_normal_(self.positional_encodings[i], std=0.2)
        #     nn.init.trunc_normal_(self.positional_encodings2[i], std=0.2)

            # self.cas[i] = LoFTREncoderLayer(num_ch_enc[i], 8)
            self.convs[('down',i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)

        self.addition = nn.ModuleList(list(self.convs.values()))
        self.position_embeddings = nn.ParameterDict(self.position_embeddings)
    def forward(self, rgb_features, depth_features,inputs):

        final_features = []
        for i in range(len(rgb_features)):
            rgb_feature = rgb_features[i]
            depth_feature = depth_features[i]

            rgb_feature_down = self.convs[('down',i)](rgb_feature)
            depth_features_low = torch.nn.functional.interpolate(depth_feature, size=(rgb_feature_down.shape[2], rgb_feature_down.shape[3]), mode='nearest')
            low_tof_mask = inputs[('additional', 0)]['mask'].reshape(-1,self.args.zone_num,self.args.zone_num)#4,64
            resized_tof_mask = torch.nn.functional.interpolate(low_tof_mask.unsqueeze(1).float(), size=(rgb_feature_down.shape[2], rgb_feature_down.shape[3]), mode='nearest')
            resized_tof_mask_flatten = resized_tof_mask.reshape(-1,rgb_feature_down.shape[2]*rgb_feature_down.shape[3])
            rgb_feature_down_flatten = rearrange(rgb_feature_down, 'b c h w -> b (h w) c')
            rgb_feature_down_flatten = rgb_feature_down_flatten + self.position_embeddings[str(i)]
            depth_feature_low_flatten = rearrange(depth_features_low, 'b c h w -> b (h w) c')
            rgb_affine = torch.matmul(rgb_feature_down_flatten, rgb_feature_down_flatten.transpose(1, 2))
            rgb_affine[resized_tof_mask_flatten.unsqueeze(1).repeat(1,rgb_affine.shape[-1],1) == 0.] =  -1e9
            rgb_affine = torch.nn.functional.softmax(rgb_affine, dim=2)
            depth_new_flatten = torch.matmul(rgb_affine, depth_feature_low_flatten)
            depth_new = rearrange(depth_new_flatten, 'b (h w) c -> b c h w', h=rgb_feature_down.shape[2], w=rgb_feature_down.shape[3])
            depth_new = torch.nn.functional.interpolate(depth_new, size=(rgb_feature.shape[2], rgb_feature.shape[3]), mode='nearest')
            final_features.append(depth_new+rgb_feature)

        return final_features