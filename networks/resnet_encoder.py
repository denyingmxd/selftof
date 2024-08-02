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

    def forward(self, input_image,depth_image=None,index=None):
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))  #64, 144x192
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1]))) #64, 72x96
        features.append(self.encoder.layer2(features[-1])) #128, 36x48
        features.append(self.encoder.layer3(features[-1])) #256, 18x24
        features.append(self.encoder.layer4(features[-1])) #512, 9x12

        # return features,[torch.zeros_like(features[i]) for i in range(5)]
        return features

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



class DepthEncoder(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder, self).__init__()
        self.num_ch_enc = num_ch_enc

        self.pre_conv = convbnrelu(d_num*num_input_images, num_ch_enc[0], 7, 2, 3)
        self.conv1 = nn.Sequential(
            convbnrelu(num_ch_enc[0], num_ch_enc[1], 3, 2, 1),
            ConvBlock(num_ch_enc[1], num_ch_enc[1])
        )
        self.conv2 = nn.Sequential(
            convbnrelu(num_ch_enc[1], num_ch_enc[2], 3, 2, 1),
            ConvBlock(num_ch_enc[2], num_ch_enc[2])
        )
        self.conv3 = nn.Sequential(
            convbnrelu(num_ch_enc[2], num_ch_enc[3], 3, 2, 1),
            ConvBlock(num_ch_enc[3], num_ch_enc[3])
        )
        self.conv4 = nn.Sequential(
            convbnrelu(num_ch_enc[3], num_ch_enc[4], 3, 2, 1),
            ConvBlock(num_ch_enc[4], num_ch_enc[4])
        )


    def forward(self, input_image):
        features = []
        x = self.pre_conv(input_image)
        features.append(x)
        features.append(self.conv1(features[-1]))
        features.append(self.conv2(features[-1]))
        features.append(self.conv3(features[-1]))
        features.append(self.conv4(features[-1]))
        return features




class DepthEncoder_6(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_6, self).__init__()
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


    def forward(self, input_image):

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


class DepthEncoder_7(nn.Module):
    def __init__(self,num_ch_enc,d_num,num_input_images):
        super(DepthEncoder_7, self).__init__()
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


    def forward(self, input_image):

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







class RGBD_Encoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1,args=None):
        super(RGBD_Encoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.rgb_encoder = ResnetEncoder(num_layers, pretrained, num_input_images)
        if 'multi' in args.sparse_depth_input_type:
            d_num = int(args.sparse_depth_input_type.split('_')[-1])
        else:
            d_num = 1
        if args.encoder_type==0:
            self.depth_encoder = DepthEncoder(self.num_ch_enc, d_num,num_input_images)
        else:
            functions = {name: obj for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__}
            self.depth_encoder = functions["DepthEncoder_{}".format(args.encoder_type)](self.num_ch_enc, d_num,num_input_images)



        self.args = args


    def forward(self, rgb_image, inputs, index=0):
        rgb_features = self.rgb_encoder(rgb_image)
        if index==0:
            depth_features = self.depth_encoder(inputs[('tof_depth',0)])
        else:
            depth_inputs = torch.cat([inputs[('tof_depth',0)],inputs[('tof_depth',index)]],1)
            depth_features = self.depth_encoder(depth_inputs)
        features =[rgb_features[i]+depth_features[i] for i in range(len(rgb_features))]

        return features,rgb_features