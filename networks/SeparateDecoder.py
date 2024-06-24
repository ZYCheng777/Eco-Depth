from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from .hr_layers import *

from networks.scale_casa import scale_casa_HAM, scale_casa
from networks.fusion_decoder_module import DecoderModule



class SFusionDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(SFusionDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc


        # decoder
        self.convs = OrderedDict()

        self.fusion_decoder = DecoderModule([24, 24, 40, 64])


        self.convs[("parallel_conv"), 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])

        self.convs[("fusionAtt", 0, 0)] = fusionModule(int(self.num_ch_enc[0] * 0.5), int(self.num_ch_enc[0] * 0.5))

        self.convs[("attention", 1)] = fSEModule(self.num_ch_enc[0], self.num_ch_enc[0])

        self.convs[("dispconv", 0)] = Conv3x3(48, self.num_output_channels)

        self.CASA = scale_casa_HAM([48, 48, 80, 128])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_feature):
        self.outputs = {}
        self.decoder_imfeatures = []
        self.decoder_subimfeatures = []

        input_features = self.CASA(input_feature)

        e0 = input_features[0]

        x0 = torch.chunk(input_features[1], 2, dim=1)
        x1 = torch.chunk(input_features[2], 2, dim=1)
        x2 = torch.chunk(input_features[3], 2, dim=1)
        x3 = torch.chunk(input_features[4], 2, dim=1)

        im1 = x0[0]
        im2 = x1[0]
        im3 = x2[0]
        im4 = x3[0]

        subim1 = x0[1]
        subim2 = x1[1]
        subim3 = x2[1]
        subim4 = x3[1]
        self.decoder_imfeatures.append(im1)
        self.decoder_imfeatures.append(im2)
        self.decoder_imfeatures.append(im3)
        self.decoder_imfeatures.append(im4)


        im_feat = self.fusion_decoder(self.decoder_imfeatures)

        self.decoder_subimfeatures.append(subim1)
        self.decoder_subimfeatures.append(subim2)
        self.decoder_subimfeatures.append(subim3)
        self.decoder_subimfeatures.append(subim4)

        subim_feat = self.fusion_decoder(self.decoder_subimfeatures)


        feat = self.convs[("fusionAtt", 0, 0)](torch.cat([im_feat, subim_feat], dim=1))

        e0 = self.convs[("parallel_conv"), 0](e0)

        out = self.convs[("attention", 1)](feat, e0)

        out = self.convs[("parallel_conv"), 0](out)

        out = updown_sample(out, 2)

        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](out))

        return self.outputs  # single-scale depth