# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


# depth_decoder
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from .hr_layers import *
#from scale_casa import scale_casa, scale_casa_HAM, CASA, scale_casa_HAM_T




class DecoderModule(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=1):
        super(DecoderModule, self).__init__()

        self.num_output_channels = num_output_channels


        self.num_ch_enc = num_ch_enc


        # decoder
        self.convs = OrderedDict()

        self.convs[("parallel_conv"), 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])


        self.convs[("conv1x1", 2_1)] = ConvBlock1x1(self.num_ch_enc[1] + self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("conv1x1", 3_2)] = ConvBlock1x1(self.num_ch_enc[2] + self.num_ch_enc[1], self.num_ch_enc[1])


        self.convs[("attention", 3)] = fSEModule(self.num_ch_enc[2], self.num_ch_enc[3])

        self.convs[("attention", 2)] = fSEModule(self.num_ch_enc[1], self.num_ch_enc[2])

        self.convs[("attention", 1)] = fSEModule_eq(self.num_ch_enc[0], self.num_ch_enc[1])

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def FusionConv(self, conv, high_feature, low_feature):
        high_features = [upsample(high_feature)]
        high_features.append(low_feature)
        high_features = torch.cat(high_features, 1)


        return conv(high_features)

    def FusionConv_eq(self, conv, high_feature, low_feature):
        high_features = [high_feature]
        high_features.append(low_feature)
        high_features = torch.cat(high_features, 1)

        return conv(high_features)

    def forward(self, input_features):
        e3 = input_features[3]
        e2 = input_features[2]
        e1 = input_features[1]
        e0 = input_features[0]

        d1_1 = self.convs[("parallel_conv"), 0](e0)
        d1_2 = self.convs[("parallel_conv"), 1](e1)
        d1_3 = self.convs[("parallel_conv"), 2](e2)

        d14_3 = self.convs[("attention", 3)](e3, d1_3)
        d13_2 = self.FusionConv(self.convs[("conv1x1", 3_2)], d1_3, d1_2)
        d12_1 = self.FusionConv_eq(self.convs[("conv1x1", 2_1)], d1_2, d1_1)

        d2_1 = self.convs[("parallel_conv"), 0](d12_1)
        d2_2 = self.convs[("parallel_conv"), 1](d13_2)

        d23_2 = self.convs[("attention", 2)](d14_3, d2_2)
        d22_1 = self.FusionConv_eq(self.convs[("conv1x1", 2_1)], d2_2, d2_1)


        d3_0 = self.convs[("parallel_conv"), 0](d22_1)
        d32_1 = self.convs[("attention", 1)](d23_2, d3_0)

        d = self.convs[("parallel_conv"), 0](d32_1)


        return d






