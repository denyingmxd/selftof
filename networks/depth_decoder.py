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
        if args.addition_type>0:
            addition_network = getattr(layers, "AdditionNetwork_{}".format(args.addition_type))
            self.addition_network = addition_network(num_ch_enc,args)
            # if args.adpative_scale_type==1 or args.adpative_scale_type==3 :
            #     self.convs[("adpative_scale", s)] = nn.Sequential(
            #         Conv3x3(self.num_ch_dec[s], args.num_scale_banks),
            #         nn.Softmax(dim=1)
            #     )
            # elif args.adpative_scale_type==2 or args.adpative_scale_type==4 or args.adpative_scale_type==6 or args.adpative_scale_type==5:
            #     self.convs[("adpative_scale", s)] = nn.Sequential(
            #         Conv3x3(self.num_ch_enc[s], args.num_scale_banks),
            #         nn.Softmax(dim=1)
            #     )
            # elif args.adpative_scale_type ==7 or args.adpative_scale_type==8:
            #     self.convs[("adpative_scale", s)] = nn.Sequential(
            #         Conv3x3(self.num_ch_enc[s] if args.adpative_scale_type ==8 else self.num_ch_dec[s], 4+1),
            #         nn.Softmax(dim=1)
            #     )
            #
            # elif args.adpative_scale_type ==9 or args.adpative_scale_type==10:
            #     self.convs[("adpative_scale", s)] = nn.Sequential(
            #         Conv3x3(self.num_ch_enc[s] if args.adpative_scale_type ==10 else self.num_ch_dec[s], 8+1),
            #         nn.Softmax(dim=1)
            #     )
            #
            # if args.global_scale_type==1 or args.global_scale_type==2:
            #     self.convs[("flatten", s)] = nn.AdaptiveAvgPool2d((1, 1))
            #     self.convs[('pwconv1',s)] = nn.Linear(self.num_ch_enc[-1], 64)  # pointwise/1x1 convs, implemented with linear layers
            #     self.convs[('act',s)] = nn.GELU()
            #     self.convs[('pwconv2',s)] = nn.Linear(64, 64)
            #     self.convs[('norm',s)] = nn.LayerNorm(self.num_ch_enc[-1])

            # elif args.global_scale_type==3 or args.global_scale_type==4:
            #     self.convs[("flatten", s)] = nn.AdaptiveAvgPool2d((1, 1))
            #     self.convs[('pwconv1', s)] = nn.Linear(self.num_ch_enc[-1],64)
            #     self.convs[('act', s)] = nn.GELU()
            #     self.convs[('pwconv2', s)] = nn.Linear(64, 64)
            #     self.convs[('norm', s)] = nn.LayerNorm(self.num_ch_enc[-1])

            if args.flow_type==1:
                self.convs[("flow", s)] = AlignedModule(self.num_ch_enc[0], self.num_ch_enc[0]//2)
            elif args.flow_type==2:
                self.convs[("flow", s)] = AlignedModule(self.num_ch_enc[0], self.num_ch_enc[0]//2)
            elif args.flow_type==3:
                self.convs[("flow", s)] = AlignedModule(self.num_ch_dec[0], self.num_ch_dec[0]//2)
            elif args.flow_type==4:
                self.convs[("flow", s)] = AlignedModule2((self.num_ch_enc[0]), (self.num_ch_dec[0]))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_features, norm_pix_coords, inputs, args,rgb_features):
        #
        outputs = {}

        if args.addition_type==0:
            pass
        else:
            tof_features = [input_features[i] - rgb_features[i] for i in range(len(input_features))]

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
                    if self.PixelCoorModu:
                        len_feat = torch.norm(feat, p=2, dim=1, keepdim=True)

                        len_norm_pix_coords = torch.norm(norm_pix_coords[i], p=2, dim=1, keepdim=True)
                        coeff = feat + norm_pix_coords[i]*len_feat/len_norm_pix_coords

                        outputs[("coeff", i)] = coeff
                        outputs[("disp", i)] = (coeff * norm_pix_coords[i]).sum(1, keepdim=True)

                    else:
                        outputs[("coeff", i)] = feat
                        outputs[("disp", i)] = F.relu((feat * norm_pix_coords[i]).sum(1, keepdim=True), inplace=True)

                do_scale = args.scale_by_tof_disp + args.scale_by_tof_disp_all + (args.fuse_scale_type>0) + (args.scale_by_tof_depth_type>0)
                assert do_scale <= 1, "Only one scale method can be used at a time"

                do_adpative_scale = args.adpative_scale_type>0
                assert do_adpative_scale <= 1, "Only one scale method can be used at a time"

                do_global_scale = args.global_scale_type>0

                assert do_scale + do_adpative_scale + do_global_scale<= 1, "Only one scale method can be used at a time"

                if do_scale:
                    outputs[("disp", i)] = self.scale_transform(outputs,i,inputs, args)

                # if do_adpative_scale:
                #     raw_scale_maps = self.get_scale_maps(outputs, i, inputs, args)
                #     if args.adpative_scale_type == 1 or args.adpative_scale_type == 3 or \
                #         args.adpative_scale_type == 7 or args.adpative_scale_type == 9 :
                #         weights_scale_map = self.convs[("adpative_scale", i)](x)
                #     elif args.adpative_scale_type == 2 or args.adpative_scale_type == 4 or\
                #             args.adpative_scale_type == 5 or args.adpative_scale_type == 6 or\
                #             args.adpative_scale_type == 8 or args.adpative_scale_type == 10:
                #         weights_scale_map = self.convs[("adpative_scale", i)](upsample(rgb_features[i]))
                #
                #
                #     final_scale_map = torch.sum(raw_scale_maps * weights_scale_map, dim=1, keepdim=True)
                #     outputs[('scale_map',i)] = final_scale_map
                #     output_depth = 1 / (1e-7 + outputs[("disp", i)])
                #
                #     scaled_output_depth = output_depth * final_scale_map
                #     outputs[("disp", i)] = 1 / (1e-7 + scaled_output_depth)
                #
                # if do_global_scale:
                #     final_depth = self.get_global_depth(inputs, args,outputs,i,input_features,rgb_features)
                #
                #     outputs[("disp", i)] = 1 / (final_depth)
                #
                # do_flow = args.flow_type>0
                #
                # if do_flow:
                #     outputs[("disp", i)] = self.flow_transform(outputs,i,inputs, args,input_features,rgb_features,x)



        return outputs

    def flow_transform(self,outputs,i,inputs,args,input_features,rgb_features,feat):
        depth = 1 / (outputs[("disp", i)])
        B, C, H, W = depth.shape
        if args.flow_type==1:
            features = input_features[i]
            flow,refined_depth = self.convs[("flow", i)](features,depth)
        elif args.flow_type==2:
            features = rgb_features[i]
            flow,refined_depth = self.convs[("flow", i)](features,depth)
        elif args.flow_type==3:
            features = feat
            flow,refined_depth = self.convs[("flow", i)](features, depth)
        elif args.flow_type==4:
            features1, features2 = rgb_features[i], feat
            flow,refined_depth = self.convs[("flow", i)](features1, features2, depth)

        outputs[('flow',i)] = flow
        return 1./refined_depth

    def scale_transform(self,outputs,i,inputs,args):
        if args.scale_by_tof_disp:
            scaled_disp = self.scale_zone(outputs, i, inputs, args)

        if args.scale_by_tof_disp_all:
            scaled_disp = self.scale_all(outputs, i, inputs, args)


        if args.scale_by_tof_depth_type > 0:
            depth = 1 / (1e-7 + outputs[("disp", i)])
            scaled_disp = self.scale_tof_depth(depth, i, inputs, args,outputs)

        return scaled_disp

    #
    # def scale_zone(self,outputs,i,inputs,args):
    #     B, C, H, W = outputs[("disp", i)].shape
    #     output_disp = outputs[("disp", i)]
    #     aa, bb, cc, dd = 0, 0, H, W
    #     n = args.zone_num
    #     p1, p2 = torch.div(cc - aa, n, rounding_mode='floor'), torch.div(dd - bb, n, rounding_mode='floor')
    #     hist_data = inputs[('additional', 0)]['hist_data'].to(output_disp.device)
    #     cropped_output_disp = output_disp[:, :, aa:cc, bb:dd]
    #     folded_output_disp = rearrange(cropped_output_disp,
    #                                    'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)', zn1=n, zn2=n,
    #                                    p1=p1, p2=p2)
    #
    #     folded_output_disp *= (
    #             (1. / hist_data[:, :, 0:1]).unsqueeze(1) /
    #             torch.median(folded_output_disp, dim=-1, keepdim=True).values)
    #     folded_output_disp = rearrange(folded_output_disp,
    #                                    'b c (zn1 zn2) (p1 p2) -> b c (zn1 p1) (zn2 p2)', zn1=n, zn2=n,
    #                                    p1=p1, p2=p2)
    #     return folded_output_disp
    #
    # def scale_all(self,outputs,i,inputs,args):
    #     B, C, H, W = outputs[("disp", i)].shape
    #     output_disp = outputs[("disp", i)]
    #     output_disp_flatten = outputs[("disp", i)].reshape(B, -1)
    #     hist_data = inputs[('additional', 0)]['hist_data'].to(output_disp.device)
    #
    #     ratios = torch.median((1. / hist_data[:, :, 0]), dim=-1, keepdim=True).values \
    #              / torch.median(output_disp_flatten, dim=-1, keepdim=True).values
    #     return output_disp * ratios.unsqueeze(-1).unsqueeze(-1)


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
        std = folded_output_depth.std(dim=-1, keepdim=True) +1e-7
        median = torch.median(folded_output_depth, dim=-1, keepdim=True).values


        if args.scale_by_tof_depth_type == 1:
            ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)
            outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
            folded_output_depth *= ratios


        elif args.scale_by_tof_depth_type == 2:
            if args.oracle and args.drop_rate==0.0:
                output_depth_flatten = folded_output_depth.reshape(B, -1)

                ratios = (torch.median(hist_data[:, :, 0], dim=-1, keepdim=True).values /
                                        torch.median(output_depth_flatten, dim=-1, keepdim=True).values)
                outputs[('global_ratio', i)] = ratios.unsqueeze(-1).unsqueeze(-1)
                folded_output_depth *= ratios.unsqueeze(-1).unsqueeze(-1)
            else:
                tof_mask = inputs[('additional', 0)]['tof_mask']
                hist_mask = inputs[('additional', 0)]['mask']
                ratios = [torch.median(hist_data[i,:,0][hist_mask[i].bool()]) / torch.median(cropped_output_depth[i][tof_mask[i].bool()]) for i in range(len(folded_output_depth))]
                ratios = torch.stack(ratios).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                folded_output_depth *= ratios

        elif args.scale_by_tof_depth_type == 3:
            folded_output_depth = (folded_output_depth - mean) / std * (hist_data[:, :, 1:2]+1e-7).unsqueeze(1) + hist_data[:, :, 0:1].unsqueeze(1)

        elif args.scale_by_tof_depth_type == 4:
            folded_output_depth = (folded_output_depth - mean) + hist_data[:, :, 0:1].unsqueeze(1)

        elif args.scale_by_tof_depth_type == 5:
            folded_output_depth = (folded_output_depth - median) / std * (hist_data[:, :, 1:2]+1e-7).unsqueeze(1) + hist_data[:, :, 0:1].unsqueeze(1)

        elif args.scale_by_tof_depth_type == 6:
            folded_output_depth = (folded_output_depth - median) + hist_data[:, :, 0:1].unsqueeze(1)

        elif args.scale_by_tof_depth_type == 7:
            folded_output_depth *= ((hist_data[:, :, 0:1]).unsqueeze(1) / mean)

        elif args.scale_by_tof_depth_type == 8:
            if args.oracle and args.drop_rate==0.0:
                ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)
                outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
                ratios = torch.median(ratios,dim=2,keepdim=True).values
                outputs[('global_ratio', i)] = ratios
                folded_output_depth *= ratios
            else:
                ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)
                outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
                hist_mask = inputs[('additional', 0)]['mask'].unsqueeze(1).unsqueeze(-1)
                ratios = [torch.median(ratios[i][hist_mask[i].bool()]) for i in range(len(ratios))]
                ratios = torch.stack(ratios).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                outputs[('global_ratio', i)] = ratios
                folded_output_depth *= ratios

        elif args.scale_by_tof_depth_type == 9:
            ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)

            ratios = torch.mean(ratios,dim=2,keepdim=True)
            folded_output_depth *= ratios

        elif args.scale_by_tof_depth_type == 10:
            median = torch.quantile(folded_output_depth, 0.5, dim=-1, keepdim=True)
            ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)
            outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
            folded_output_depth *= ratios

        elif args.scale_by_tof_depth_type == 11:
            median = torch.quantile(folded_output_depth, 0.5, dim=-1, keepdim=True)
            ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)
            outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
            ratios = torch.quantile(ratios,0.5,dim=2,keepdim=True)
            outputs[('global_ratio', i)] = ratios
            folded_output_depth *= ratios

        # elif args.scale_by_tof_depth_type == 12:
        #     if args.oracle:
        #         pixel_distributions = [get_hist_parallel_with_torch_tracking(output_depth[i], args) for i in range(output_depth.shape[0])]
        #         pixel_distributions = torch.stack(pixel_distributions,0)
        #         outputs[('pixel_distributions',i)] = pixel_distributions
        #         pixel_distributions = rearrange(pixel_distributions, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)', zn1=n, zn2=n,
        #                   p1=p1, p2=p2).contiguous()
        #         flatten_depth = rearrange(folded_output_depth, 'b c (zn1 zn2) (p1 p2) -> (b c zn1 zn2) (p1 p2)', zn1=n, zn2=n,p1=p1, p2=p2)
        #         flatten_pixel_distributions = rearrange(pixel_distributions, 'b c (zn1 zn2) (p1 p2) -> (b c zn1 zn2) (p1 p2)', zn1=n, zn2=n,p1=p1, p2=p2)
        #         refined_mean = [torch.mean(flatten_depth[i][flatten_pixel_distributions[i]]) for i in range(len(flatten_depth))]
        #         refined_mean = torch.stack(refined_mean,0)
        #         refined_mean = rearrange(refined_mean, '(b c zn1 zn2) -> b c (zn1 zn2)', zn1=n, zn2=n, b=B, c=C).unsqueeze(-1)
        #         ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / refined_mean)
        #
        #         outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
        #         ratios = torch.median(ratios, dim=2, keepdim=True).values
        #         outputs[('global_ratio', i)] = ratios
        #         folded_output_depth *= ratios
        #
        #     else:
        #         raise NotImplementedError
        #
        # elif args.scale_by_tof_depth_type == 13:
        #     if args.oracle:
        #         pixel_distributions = [get_hist_parallel_with_torch_tracking(output_depth[i], args) for i in range(output_depth.shape[0])]
        #         pixel_distributions = torch.stack(pixel_distributions,0)
        #         outputs[('pixel_distributions',i)] = pixel_distributions
        #         pixel_distributions = rearrange(pixel_distributions, 'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)', zn1=n, zn2=n,
        #                   p1=p1, p2=p2).contiguous()
        #         flatten_depth = rearrange(folded_output_depth, 'b c (zn1 zn2) (p1 p2) -> (b c zn1 zn2) (p1 p2)', zn1=n, zn2=n,p1=p1, p2=p2)
        #         flatten_pixel_distributions = rearrange(pixel_distributions, 'b c (zn1 zn2) (p1 p2) -> (b c zn1 zn2) (p1 p2)', zn1=n, zn2=n,p1=p1, p2=p2)
        #         refined_mean = [torch.median(flatten_depth[i][flatten_pixel_distributions[i]]) for i in range(len(flatten_depth))]
        #         refined_mean = torch.stack(refined_mean,0)
        #         refined_mean = rearrange(refined_mean, '(b c zn1 zn2) -> b c (zn1 zn2)', zn1=n, zn2=n, b=B, c=C).unsqueeze(-1)
        #         ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / refined_mean)
        #
        #         outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
        #         ratios = torch.median(ratios, dim=2, keepdim=True).values
        #         outputs[('global_ratio', i)] = ratios
        #         folded_output_depth *= ratios
        #
        #     else:
        #         raise NotImplementedError


        folded_output_depth = rearrange(folded_output_depth,
                                       'b c (zn1 zn2) (p1 p2) -> b c (zn1 p1) (zn2 p2)', zn1=n, zn2=n,
                                       p1=p1, p2=p2)


        return 1./folded_output_depth


    # def get_scale_maps(self,outputs,i,inputs,args):
    #     depth = 1 / (1e-7 + outputs[("disp", i)])
    #     B, C, H, W = depth.shape
    #     output_depth = depth
    #     aa, bb, cc, dd = 0, 0, H, W
    #     n = args.zone_num
    #     p1, p2 = torch.div(cc - aa, n, rounding_mode='floor'), torch.div(dd - bb, n, rounding_mode='floor')
    #     hist_data = inputs[('additional', 0)]['hist_data'].to(output_depth.device)
    #     cropped_output_depth = output_depth[:, :, aa:cc, bb:dd]
    #     folded_output_depth = rearrange(cropped_output_depth,
    #                                     'b c (zn1 p1) (zn2 p2) -> b c (zn1 zn2) (p1 p2)', zn1=n, zn2=n,
    #                                     p1=p1, p2=p2)
    #
    #     median = torch.median(folded_output_depth, dim=-1).values
    #
    #
    #     if args.adpative_scale_type == 1 or args.adpative_scale_type == 2 or args.adpative_scale_type == 5 or args.adpative_scale_type == 6:
    #         delta = 1e-3
    #         sample_ppf = torch.Tensor(np.arange(delta, 1, (1 - 2 * delta) / (args.num_scale_banks - 1)).tolist()).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1).to(median.device)
    #         d = torch.distributions.Normal(hist_data[..., 0:1], hist_data[..., 1:2])
    #         fh = d.icdf(sample_ppf).to(median.device).permute(0, 2, 1)
    #         ratios = fh / median
    #
    #     if args.adpative_scale_type == 3 or args.adpative_scale_type == 4:
    #
    #         d = torch.distributions.Normal(hist_data[..., 0], hist_data[..., 1])
    #         sample_easy = d.sample((args.num_scale_banks,)).permute(1, 0, 2)
    #         ratios = sample_easy / median
    #
    #     if args.adpative_scale_type == 7 or args.adpative_scale_type == 8 or \
    #         args.adpative_scale_type == 9 or args.adpative_scale_type == 10:
    #         ratios = hist_data[..., 0].unsqueeze(1) / median
    #
    #     ratios = rearrange(ratios, 'b c (zn1 zn2) -> b c zn1 zn2', zn1=n, zn2=n)
    #     if  args.adpative_scale_type <= 4:
    #         ratios  = F.interpolate(ratios, size=(H, W), mode='nearest')
    #     elif args.adpative_scale_type == 5:
    #         ratios = F.interpolate(ratios, size=(H, W), mode='bilinear', align_corners=True)
    #     elif args.adpative_scale_type == 6:
    #         ratios = F.interpolate(ratios, size=(H, W), mode='bilinear', align_corners=False)
    #     elif args.adpative_scale_type == 7 or args.adpative_scale_type == 8:
    #         neighbour_ratios = get_4_neighbors(ratios)
    #         neighbour_ratios = neighbour_ratios.squeeze(1).permute(0,3,2,1)
    #         ratios = torch.cat([ratios, neighbour_ratios], dim=1)
    #         ratios = F.interpolate(ratios, size=(H, W), mode='nearest')
    #
    #     elif args.adpative_scale_type == 9 or args.adpative_scale_type == 10:
    #         neighbour_ratios = get_8_neighbors(ratios)
    #         neighbour_ratios = neighbour_ratios.squeeze(1).permute(0,3,2,1)
    #         ratios = torch.cat([ratios, neighbour_ratios], dim=1)
    #         ratios = F.interpolate(ratios, size=(H, W), mode='nearest')
    #
    #
    #     if args.detach_scale:
    #         ratios = ratios.detach().clone()
    #
    #     return ratios
    #
    def get_global_depth(self,inputs, args,outputs,i,input_features,rgb_features):
        depth = 1 / (1e-7 + outputs[("disp", i)])
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
        std = folded_output_depth.std(dim=-1, keepdim=True) + 1e-7
        median = torch.median(folded_output_depth, dim=-1, keepdim=True).values

        if args.global_scale_type == 1 or args.global_scale_type == 2:
            ratios = ((hist_data[:, :, 0:1]).unsqueeze(1) / median)
            outputs[('scale_map', i)] = rearrange(ratios, 'b c (zn1 zn2) 1 -> b c zn1 zn2', zn1=n, zn2=n)
            ratios = ratios.permute(0,2,1,3)
            features = input_features[-1] if args.global_scale_type == 1 else rgb_features[-1]
            features = self.convs[("flatten", i)](features)
            features = features.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            features = self.convs[('norm',i)](features)
            features = self.convs[('pwconv1',i)](features)
            features = self.convs[('act',i)](features)
            features = self.convs[('pwconv2',i)](features)
            features = features.permute(0, 3, 1, 2)  #
            ratio_weights = nn.Softmax(dim=1)(features)
            global_ratio = (ratios*ratio_weights).sum(1,keepdim=True)
            folded_output_depth *= global_ratio
            outputs[('global_ratio',i)] = global_ratio



        folded_output_depth = rearrange(folded_output_depth,
                                       'b c (zn1 zn2) (p1 p2) -> b c (zn1 p1) (zn2 p2)', zn1=n, zn2=n,
                                       p1=p1, p2=p2)

        return folded_output_depth


