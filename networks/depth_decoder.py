# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from layers import *
import layers





class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), depth_repr='plane', use_skips=True, PixelCoorModu=True,args=None):
        super(DepthDecoder, self).__init__()
        if depth_repr == 'plane':
            num_output_channels = 3
        elif depth_repr == 'disp':
            num_output_channels = 1
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.PixelCoorModu = PixelCoorModu

        # decoder
        self.convs = OrderedDict()
        if args.conv_block_type==0:
            used_conv_block = ConvBlock
        else:
            used_conv_block = layers.__dict__["ConvBlock_{}".format(args.conv_block_type)]
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = used_conv_block(num_ch_in, num_ch_out, i, 0)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = used_conv_block(num_ch_in, num_ch_out, i, 1)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        if args.depth_addition_type>0:
            addition_network = getattr(layers, "AdditionNetwork_{}".format(args.depth_addition_type))
            self.addition_network = addition_network(num_ch_enc,args)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def forward(self, rgb_features, norm_pix_coords, inputs, args,tof_features):
        #
        outputs = {}
        if args.depth_addition_type==0:
            input_features = [rgb_features[i]+tof_features[i] for i in range(len(rgb_features))]
        else:
            input_features = self.addition_network(rgb_features, tof_features, inputs)

        # # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                feat = self.convs[("dispconv", i)](x)
                if self.num_output_channels == 1:
                    outputs[("disp", i)] = self.sigmoid(feat)
                else:
                    raise NotImplementedError

                do_scale = args.scale_by_tof_depth_type>0

                if do_scale and 'tof' in args.sparse_d_type:
                    outputs[("disp", i)] = self.scale_transform(outputs,i,inputs, args)


        return outputs


    def scale_transform(self,outputs,i,inputs,args):
        if args.scale_by_tof_depth_type > 0:
            depth = 1 / (1e-7 + outputs[("disp", i)])
            scaled_disp = self.scale_tof_depth(depth, i, inputs, args,outputs)

        return scaled_disp


    def scale_tof_depth(self,depth,i,inputs,args,outputs):
        B, C, H, W = depth.shape
        output_depth = depth
        aa, bb, cc, dd = 0, 0, H, W
        n = args.zone_num
        p1, p2 = torch.div(cc - aa, n, rounding_mode='floor'), torch.div(dd - bb, n, rounding_mode='floor')
        hist_data = inputs[('additional', 0)]['hist_data'].to(output_depth.device)
        cropped_output_depth = output_depth[:, :, aa:cc, bb:dd]
        folded_output_depth = rearrange(cropped_output_depth,
                                       'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)', zn1=n, zn2=n,
                                       p1=p1, p2=p2)
        mean = folded_output_depth.mean(dim=-1, keepdim=True)

        median = torch.median(folded_output_depth, dim=-1, keepdim=True).values

        if args.scale_by_tof_depth_type == 1:
            tof_mask = inputs[('additional', 0)]['tof_mask']
            hist_mask = inputs[('additional', 0)]['mask']
            ratios = [torch.median(hist_data[i,:,0][hist_mask[i].bool()]) / torch.median(cropped_output_depth[i][tof_mask[i].bool()]) for i in range(len(folded_output_depth))]
            ratios = torch.stack(ratios).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            folded_output_depth *= ratios

        elif args.scale_by_tof_depth_type == 2:
            tof_mask = inputs[('additional', 0)]['tof_mask']
            hist_mask = inputs[('additional', 0)]['mask']
            pixel_distributions = [get_hist_parallel_with_torch_tracking(output_depth[i], args) for i in range(output_depth.shape[0])]
            pixel_distributions = torch.stack(pixel_distributions,0)
            outputs[('pixel_distributions',i)] = pixel_distributions
            combined_mask = (tof_mask.to(pixel_distributions.device))*pixel_distributions
            ratios = [torch.median(hist_data[i, :, 0][hist_mask[i].bool()]) / torch.median(cropped_output_depth[i][combined_mask[i].bool()]) for i in range(len(folded_output_depth))]
            ratios = torch.stack(ratios).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            folded_output_depth *= ratios


        elif args.scale_by_tof_depth_type == 3:
            ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)
            outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
            hist_mask = inputs[('additional', 0)]['mask'].unsqueeze(1).unsqueeze(-1)
            ratios = [torch.median(ratios[i][hist_mask[i].bool()]) for i in range(len(ratios))]
            ratios = torch.stack(ratios).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            outputs[('global_ratio', i)] = ratios
            folded_output_depth *= ratios

        elif args.scale_by_tof_depth_type == 4:
            ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)
            outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
            hist_mask = inputs[('additional', 0)]['mask'].unsqueeze(1).unsqueeze(-1)
            ratios = [torch.mean(ratios[i][hist_mask[i].bool()]) for i in range(len(ratios))]
            ratios = torch.stack(ratios).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            outputs[('global_ratio', i)] = ratios
            folded_output_depth *= ratios


        folded_output_depth = rearrange(folded_output_depth,
                                       'b c (zn1 zn2) (p1 p2) -> b c (zn1 p1) (zn2 p2)', zn1=n, zn2=n,
                                       p1=p1, p2=p2)


        return 1./folded_output_depth




