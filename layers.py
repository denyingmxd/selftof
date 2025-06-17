# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import numpy as np

import torch.nn.functional as F
import math
import spconv.pytorch as spconv
# from MinkowskiEngine import MinkowskiConvolution, MinkowskiELU
from torchvision.ops import DeformConv2d
import torch
from einops import rearrange, repeat
import torch.nn as nn


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

from some_codes.attn import *
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
        from MinkowskiEngine import MinkowskiConvolution, MinkowskiELU
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

        valid_area_tensor = torch.ones((B,1,cc-aa,dd-bb),device=sparse_depth0.device)
        if args.weighted_ls_type==1:
            valid_mask = torch.ones_like(hist_mask)
            means = folded_output_depth0.mean([3]).squeeze(1)
            stds = folded_output_depth0.std([3]).squeeze(1)

        elif args.weighted_ls_type==2:
            if ('pixel_distributions',0) in outputs.keys():
                weights_by_distance = outputs[('pixel_distributions',0)]
            else:
                weights_by_distance = [get_hist_parallel_with_torch_tracking(output_depth0[i],args) for i in range(output_depth0.shape[0])]
                weights_by_distance = torch.stack(weights_by_distance,0)

            weights_by_distance = rearrange(weights_by_distance, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)', zn1=n, zn2=n,
                      p1=p1, p2=p2).contiguous()

            weights_by_distance = weights_by_distance.float()
            weights_by_distance = weights_by_distance * hist_mask.unsqueeze(1).unsqueeze(-1).float()

            means,stds,valid_mask = get_weighted_mean_std(folded_output_depth0,weights_by_distance)
            weights_by_distance = weights_by_distance*(valid_mask.unsqueeze(1).unsqueeze(-1))

            outputs[('ls_weight_map', 0)] = rearrange(weights_by_distance, 'b 1 (zn1 zn2) (p1 p2) -> b 1 (zn1 p1) (zn2 p2)', zn1=n, zn2=n, p1=p1, p2=p2)

        else:
            raise NotImplementedError

        if args.sparse_depth_loss_type=='area_L2':
            mean_loss = torch.square(means - hist_data[:, :, 0]) * hist_mask
            std_loss = torch.square(stds - hist_data[:, :, 1]) * hist_mask


        weight_map = torch.ones(B,n**2)

        mean_loss = ((mean_loss*valid_mask*hist_mask).sum(dim=1) / (valid_mask*hist_mask).sum(dim=1)).mean()
        std_loss = ((std_loss*valid_mask*hist_mask).sum(dim=1) / (valid_mask*hist_mask).sum(dim=1)).mean()
        outputs[('valid_con_mask',0)] = (valid_mask*hist_mask).reshape(B,1,n,n)

        if torch.isnan(mean_loss).any() or torch.isnan(std_loss).any():
            print('nan')
            mean_loss = torch.tensor(0.0).to(mean_loss.device)
            std_loss = torch.tensor(0.0).to(std_loss.device)
            exit()
        outputs[('valid_area', 0)] = torch.zeros_like(sparse_depth0)
        outputs[('valid_area', 0)][:, :, aa:cc, bb:dd] = valid_area_tensor

        outputs[('con_weight_map', 0)] = rearrange(weight_map, 'b (zn1 zn2) -> b 1 (zn1) (zn2)', zn1=n, zn2=n)

        loss_sparse_depth = mean_loss + std_loss
        


    else:
        mean_loss, std_loss, loss_sparse_depth = torch.tensor(0.0).to(output_depth0.device), \
            torch.tensor(0.0).to(output_depth0.device), torch.tensor(0.0).to(output_depth0.device)



    return mean_loss, std_loss, loss_sparse_depth



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




class AdditionNetwork_9(nn.Module):
    def __init__(self, num_ch_enc, args, for_pose=False):
        super(AdditionNetwork_9, self).__init__()
        self.num_ch_enc = num_ch_enc
        #64,144,192
        #64, 72,96
        #128, 36, 48
        #256, 18, 24
        #512, 9, 12
        self.args = args
        self.convs = {}

        for i in range(len(num_ch_enc)-1,-1,-1):
            self.convs[('proj_1', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.convs[('proj_2', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.convs[('proj_3', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.register_buffer('pe_{}'.format(i), positionalencoding1d(num_ch_enc[i],64))

        self.addition = nn.ModuleList(list(self.convs.values()))
    def forward(self, rgb_features, depth_features,inputs):

        final_features = []
        for i in range(len(rgb_features)):
            rgb_feature = rgb_features[i]
            depth_feature = depth_features[i]

            rgb_feature_down = F.adaptive_avg_pool2d(rgb_feature,(8,8))
            depth_features_low = F.interpolate(depth_feature, size=(rgb_feature_down.shape[2], rgb_feature_down.shape[3]), mode='nearest')
            # low_tof_mask = inputs[('additional', 0)]['mask'].reshape(-1,self.args.zone_num,self.args.zone_num)#4,64
            # resized_tof_mask = torch.nn.functional.interpolate(low_tof_mask.unsqueeze(1).float(), size=(rgb_feature_down.shape[2], rgb_feature_down.shape[3]), mode='nearest')
            # resized_tof_mask_flatten = rearrange(resized_tof_mask, 'b c h w -> (b c) (h w)',c=1)
            rgb_feature_down_flatten = rearrange(rgb_feature_down, 'b c h w -> b (h w) c')
            depth_feature_low_flatten = rearrange(depth_features_low, 'b c h w -> b (h w) c')
            pe = getattr(self, 'pe_{}'.format(i))
            rgb_feature_down_flatten = rgb_feature_down_flatten+pe
            rgb_feature_down_flatten_proj_1 = self.convs[('proj_1', i)](rgb_feature_down_flatten)
            rgb_feature_down_flatten_proj_2 = self.convs[('proj_2', i)](rgb_feature_down_flatten)
            depth_feature_low_flatten_proj = self.convs[('proj_3', i)](depth_feature_low_flatten)

            # embed_dim = rgb_feature_down_flatten_proj_1.shape[-1]
            rgb_affine = torch.matmul(rgb_feature_down_flatten_proj_1, rgb_feature_down_flatten_proj_2.transpose(1, 2))
            # rgb_affine[resized_tof_mask_flatten.unsqueeze(1).repeat(1, rgb_affine.shape[-1], 1) == 0.] = -1e9
            # mask = (resized_tof_mask_flatten.unsqueeze(2).repeat(1, 1, rgb_affine.shape[-1]) == 1.).to(rgb_feature.device)

            rgb_affine = torch.nn.functional.softmax(rgb_affine, dim=2)
            # rgb_affine = rgb_affine * (~mask).float()
            depth_new_flatten = torch.matmul(rgb_affine, depth_feature_low_flatten_proj)
            depth_new = rearrange(depth_new_flatten, 'b (h w) c -> b c h w', h=rgb_feature_down.shape[2], w=rgb_feature_down.shape[3])
            depth_new = torch.nn.functional.interpolate(depth_new, size=(rgb_feature.shape[2], rgb_feature.shape[3]), mode='nearest')
            final_features.append(depth_new+rgb_feature+depth_feature)

        return final_features
