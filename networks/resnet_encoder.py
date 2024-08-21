# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from layers import ConvBlock,ConvBlock_Sub,ConvBlock_Sparse,ConvBlock_MK,ConvBlock1x1,ConvBlock_Sub1x1
import torch.nn.functional as F
import spconv.pytorch as spconv
import layers
from einops import rearrange
from layers import positionalencoding1d
from layers import DeformConv2d
class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)
        del self.encoder.fc
        # del self.encoder.avgpool
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image,inputs=None):
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))  #64, 144x192
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1]))) #64, 72x96
        features.append(self.encoder.layer2(features[-1])) #128, 36x48
        features.append(self.encoder.layer3(features[-1])) #256, 18x24
        features.append(self.encoder.layer4(features[-1])) #512, 9x12

        return features,[torch.zeros_like(features[i]) for i in range(len(features))]

def convbnrelu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)



def convbnrelu_Sub(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    assert stride == 1

    return spconv.SparseSequential(
        spconv.SubMConv2d(in_channels, out_channels, kernel_size, 1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )





class DepthEncoder_1(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_1, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.pre_conv = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)
        self.conv1 = nn.Sequential(
            convbnrelu(num_ch_enc[0], num_ch_enc[1], 1, 1, 0),
            ConvBlock(num_ch_enc[1], num_ch_enc[1])
        )
        self.conv2 = nn.Sequential(
            convbnrelu(num_ch_enc[1], num_ch_enc[2], 1, 1, 0),
            ConvBlock(num_ch_enc[2], num_ch_enc[2])
        )
        self.conv3 = nn.Sequential(
            convbnrelu(num_ch_enc[2], num_ch_enc[3], 1, 1, 0),
            ConvBlock(num_ch_enc[3], num_ch_enc[3])
        )
        self.conv4 = nn.Sequential(
            convbnrelu(num_ch_enc[3], num_ch_enc[4], 1, 1, 0),
            ConvBlock(num_ch_enc[4], num_ch_enc[4])
        )


    def forward(self, input_image,rgb_features=None):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []
        x = self.pre_conv(low_tof)
        features.append(x)
        features.append(self.conv1(features[-1]))
        features.append(self.conv2(features[-1]))
        features.append(self.conv3(features[-1]))
        features.append(self.conv4(features[-1]))
        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features





class DepthEncoder_2(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_2, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.pre_conv = convbnrelu_Sub(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)
        self.conv1 = nn.Sequential(
            convbnrelu_Sub(num_ch_enc[0], num_ch_enc[1], 1, 1, 0),
            ConvBlock_Sub(num_ch_enc[1], num_ch_enc[1])
        )
        self.conv2 = nn.Sequential(
            convbnrelu_Sub(num_ch_enc[1], num_ch_enc[2], 1, 1, 0),
            ConvBlock_Sub(num_ch_enc[2], num_ch_enc[2])
        )
        self.conv3 = nn.Sequential(
            convbnrelu_Sub(num_ch_enc[2], num_ch_enc[3], 1, 1, 0),
            ConvBlock_Sub(num_ch_enc[3], num_ch_enc[3])
        )
        self.conv4 = nn.Sequential(
            convbnrelu_Sub(num_ch_enc[3], num_ch_enc[4], 1, 1, 0),
            ConvBlock_Sub(num_ch_enc[4], num_ch_enc[4])
        )


    def forward(self, input_image,rgb_features=None):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        low_tof_sp = spconv.SparseConvTensor.from_dense(low_tof.reshape(b, h, w, c))
        features = []
        x = self.pre_conv(low_tof_sp)
        features.append(x)
        features.append(self.conv1(features[-1]))
        features.append(self.conv2(features[-1]))
        features.append(self.conv3(features[-1]))
        features.append(self.conv4(features[-1]))
        features = [F.interpolate(f.dense(), HWs[i]) for i, f in enumerate(features)]
        return features



class DepthEncoder_3(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_3, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('proj_1', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.convs[('proj_2', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.convs[('proj_3', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.register_buffer('pe_{}'.format(i), positionalencoding1d(num_ch_enc[i],64))

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            rgb_feature_down_flatten = rearrange(rgb_feature_down, 'b c h w -> b (h w) c')
            depth_feature_low_flatten = rearrange(depth_feature, 'b c h w -> b (h w) c')
            pe = getattr(self, 'pe_{}'.format(i))
            rgb_feature_down_flatten = rgb_feature_down_flatten + pe
            rgb_feature_down_flatten_proj_1 = self.convs[('proj_1', i)](rgb_feature_down_flatten)
            rgb_feature_down_flatten_proj_2 = self.convs[('proj_2', i)](rgb_feature_down_flatten)
            depth_feature_low_flatten_proj = self.convs[('proj_3', i)](depth_feature_low_flatten)
            embed_dim = rgb_feature_down_flatten_proj_1.shape[-1]
            rgb_affine = torch.matmul(rgb_feature_down_flatten_proj_1, rgb_feature_down_flatten_proj_2.transpose(1, 2))
            rgb_affine = torch.nn.functional.softmax(rgb_affine, dim=2)

            depth_new_flatten = torch.matmul(rgb_affine, depth_feature_low_flatten_proj)
            depth_new = rearrange(depth_new_flatten, 'b (h w) c -> b c h w', h=rgb_feature_down.shape[2], w=rgb_feature_down.shape[3])

            features.append(depth_new)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features




class DepthEncoder_4(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_4, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('proj_1', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.convs[('proj_2', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.convs[('proj_3', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.register_buffer('pe_{}'.format(i), positionalencoding1d(num_ch_enc[i],64))

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            rgb_feature_down_flatten = rearrange(rgb_feature_down, 'b c h w -> b (h w) c')
            depth_feature_low_flatten = rearrange(depth_feature, 'b c h w -> b (h w) c')
            pe = getattr(self, 'pe_{}'.format(i))
            rgb_feature_down_flatten = rgb_feature_down_flatten + pe
            rgb_feature_down_flatten_proj_1 = self.convs[('proj_1', i)](rgb_feature_down_flatten)
            rgb_feature_down_flatten_proj_2 = self.convs[('proj_2', i)](rgb_feature_down_flatten)
            depth_feature_low_flatten_proj = self.convs[('proj_3', i)](depth_feature_low_flatten)
            embed_dim = rgb_feature_down_flatten_proj_1.shape[-1]
            rgb_affine = torch.matmul(rgb_feature_down_flatten_proj_1, rgb_feature_down_flatten_proj_2.transpose(1, 2))/(embed_dim**0.5)
            rgb_affine = torch.nn.functional.softmax(rgb_affine, dim=2)

            depth_new_flatten = torch.matmul(rgb_affine, depth_feature_low_flatten_proj)
            depth_new = rearrange(depth_new_flatten, 'b (h w) c -> b c h w', h=rgb_feature_down.shape[2], w=rgb_feature_down.shape[3])

            features.append(depth_new)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features

class DepthEncoder_5(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_5, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('proj_1', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.convs[('proj_2', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.convs[('proj_3', i)] = nn.Linear(num_ch_enc[i], num_ch_enc[i], bias=False)
            self.register_buffer('pe_{}'.format(i), positionalencoding1d(num_ch_enc[i],64))

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            rgb_feature_down_flatten = rearrange(rgb_feature_down, 'b c h w -> b (h w) c')
            depth_feature_low_flatten = rearrange(depth_feature, 'b c h w -> b (h w) c')
            pe = getattr(self, 'pe_{}'.format(i))
            rgb_feature_down_flatten = rgb_feature_down_flatten + pe
            rgb_feature_down_flatten_proj_1 = self.convs[('proj_1', i)](rgb_feature_down_flatten)
            rgb_feature_down_flatten_proj_2 = self.convs[('proj_2', i)](rgb_feature_down_flatten)
            depth_feature_low_flatten_proj = self.convs[('proj_3', i)](depth_feature_low_flatten)
            embed_dim = rgb_feature_down_flatten_proj_1.shape[-1]
            rgb_affine = torch.matmul(rgb_feature_down_flatten_proj_1, rgb_feature_down_flatten_proj_2.transpose(1, 2))
            rgb_affine = torch.nn.functional.softmax(rgb_affine, dim=2)

            depth_new_flatten = torch.matmul(rgb_affine, depth_feature_low_flatten_proj)
            depth_new = rearrange(depth_new_flatten, 'b (h w) c -> b c h w', h=rgb_feature_down.shape[2], w=rgb_feature_down.shape[3])

            features.append(depth_new+depth_feature)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features


class DepthEncoder_6(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_6, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('combine_conv', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=3,stride=1,padding=1,bias=False)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            depth_new = self.convs[('combine_conv',i)](depth_feature+rgb_feature_down)


            features.append(depth_new)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features


class DepthEncoder_7(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_7, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('combine_conv', i)] = nn.Conv2d(2*num_ch_enc[i],num_ch_enc[i],kernel_size=3,stride=1,padding=1,bias=False)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            depth_new = self.convs[('combine_conv',i)](torch.cat((rgb_feature_down,depth_feature),dim=1))



            features.append(depth_new)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features



class DepthEncoder_8(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_8, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('combine_conv', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=3,stride=1,padding=1,bias=False)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            depth_new = self.convs[('combine_conv',i)](depth_feature+rgb_feature_down)


            features.append(depth_new+depth_feature)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features


class DepthEncoder_9(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_9, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        kernel_size = 3
        padding = kernel_size // 2
        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('offset_conv', i)] = nn.Conv2d(num_ch_enc[i], 2 * kernel_size * kernel_size,kernel_size=kernel_size, padding=padding, bias=True)
            nn.init.constant_(self.convs[('offset_conv', i)].weight, 0.)
            nn.init.constant_(self.convs[('offset_conv', i)].bias, 0.)
            self.convs[('deform_conv', i)] = DeformConv2d(num_ch_enc[i], num_ch_enc[i], kernel_size=kernel_size,padding=padding, bias=True)
            nn.init.constant_(self.convs[('deform_conv', i)].weight, 0.)
            nn.init.constant_(self.convs[('deform_conv', i)].bias, 0.)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            offset = self.convs[('offset_conv', i)](rgb_feature_down)
            h, w = rgb_feature_down.shape[2:]
            max_offset = max(h, w) / 4.
            offset = offset.clamp(-max_offset, max_offset)
            depth_new = self.convs[('deform_conv', i)](depth_feature, offset)

            features.append(depth_new+depth_feature)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features


class DepthEncoder_10(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_10, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        kernel_size = 3
        padding = kernel_size // 2
        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('offset_conv', i)] = nn.Conv2d(num_ch_enc[i], 2 * kernel_size * kernel_size,kernel_size=kernel_size, padding=padding, bias=True)
            nn.init.constant_(self.convs[('offset_conv', i)].weight, 0.)
            nn.init.constant_(self.convs[('offset_conv', i)].bias, 0.)
            self.convs[('deform_conv', i)] = DeformConv2d(num_ch_enc[i], num_ch_enc[i], kernel_size=kernel_size,padding=padding, bias=True)
            nn.init.constant_(self.convs[('deform_conv', i)].weight, 0.)
            nn.init.constant_(self.convs[('deform_conv', i)].bias, 0.)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            offset = self.convs[('offset_conv', i)](rgb_feature_down)
            h, w = rgb_feature_down.shape[2:]
            max_offset = max(h, w) / 4.
            offset = offset.clamp(-max_offset, max_offset)
            depth_new = self.convs[('deform_conv', i)](depth_feature, offset)

            features.append(depth_new)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features


class DepthEncoder_11(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_11, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.convs = {}
        self.convs[('depth_conv',0)] = convbnrelu(d_num*num_input_images, num_ch_enc[0], 1, 1, 0)


        for i in range(1,len(num_ch_enc)):
            self.convs[('depth_conv',i)] = nn.Sequential(
                convbnrelu(num_ch_enc[i-1], num_ch_enc[i], 1, 1, 0),
                ConvBlock(num_ch_enc[i], num_ch_enc[i])
            )

        kernel_size = 5
        padding = kernel_size // 2
        for i in range(len(num_ch_enc)):
            self.convs[('down', i)] = nn.Conv2d(num_ch_enc[i],num_ch_enc[i],kernel_size=2**(4-i),stride=2**(4-i),padding=0,groups=num_ch_enc[i],bias=False)
            self.convs[('offset_conv', i)] = nn.Conv2d(num_ch_enc[i], 2 * kernel_size * kernel_size,kernel_size=kernel_size, padding=padding, bias=True)
            nn.init.constant_(self.convs[('offset_conv', i)].weight, 0.)
            nn.init.constant_(self.convs[('offset_conv', i)].bias, 0.)
            self.convs[('deform_conv', i)] = DeformConv2d(num_ch_enc[i], num_ch_enc[i], kernel_size=kernel_size,padding=padding, bias=True)
            nn.init.constant_(self.convs[('deform_conv', i)].weight, 0.)
            nn.init.constant_(self.convs[('deform_conv', i)].bias, 0.)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_image,rgb_features):

        B, C, H, W = input_image.shape
        org_HW = input_image.shape[2:]
        HWs = [torch.Size((torch.tensor(org_HW, dtype=torch.float32) // (2 ** i)).int().tolist()) for i in range(1, 6)]
        low_tof = F.interpolate(input_image, (8, 8))
        b,c,h,w = low_tof.shape
        features = []


        for i in range(len(rgb_features)):
            if i == 0:
                x = low_tof
            else:
                x = features[-1]
            depth_feature = self.convs[('depth_conv', i)](x)
            rgb_feature = rgb_features[i]
            rgb_feature_down = self.convs[('down', i)](rgb_feature)

            offset = self.convs[('offset_conv', i)](rgb_feature_down)
            h, w = rgb_feature_down.shape[2:]
            max_offset = max(h, w) / 4.
            offset = offset.clamp(-max_offset, max_offset)
            depth_new = self.convs[('deform_conv', i)](depth_feature, offset)

            features.append(depth_new)


        features = [F.interpolate(f, HWs[i]) for i, f in enumerate(features)]
        return features



class RGBD_Encoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1,args=None):
        super(RGBD_Encoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.rgb_encoder = ResnetEncoder(num_layers, pretrained, num_input_images)
        if 'multi' in args.sparse_depth_input_type:
            d_num = int(args.sparse_depth_input_type.split('_')[-1])
        else:
            d_num = 1

        functions = {name: obj for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__}
        self.depth_encoder = functions["DepthEncoder_{}".format(args.depth_encoder_type)](self.num_ch_enc, d_num,num_input_images)



        self.args = args


    def forward(self, rgb_image, inputs):
        rgb_features = self.rgb_encoder(rgb_image)[0]

        depth_features = self.depth_encoder(inputs[('tof_depth',0)],rgb_features)

        return rgb_features,depth_features



class RGBD_Pose_Encoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1,args=None):
        super(RGBD_Pose_Encoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.rgb_encoder = ResnetEncoder(num_layers, pretrained, num_input_images)
        if 'multi' in args.sparse_depth_input_type:
            d_num = int(args.sparse_depth_input_type.split('_')[-1])
        else:
            d_num = 1

        functions = {name: obj for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__}
        self.depth_encoder = functions["DepthEncoder_{}".format(args.depth_encoder_type)](self.num_ch_enc, d_num,num_input_images)

        self.args = args


        if args.pose_addition_type>0:
            addition_network = getattr(layers, "AdditionNetwork_{}".format(args.pose_addition_type))
            self.addition_network = addition_network([self.num_ch_enc[-1]],args,for_pose=True)



    def forward(self, rgb_image, inputs,indexes):
        rgb_features = self.rgb_encoder(rgb_image)[0]

        depth_input = torch.cat([inputs[('tof_depth',i)] for i in indexes],dim=1)
        depth_features = self.depth_encoder(depth_input,rgb_features)

        if self.args.pose_addition_type>0:
            features = self.addition_network([rgb_features[-1]], [depth_features[-1]], inputs)
        else:
            features = [rgb_features[-1]+depth_features[-1]]
        return features

