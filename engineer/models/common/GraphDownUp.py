#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import itertools




class GraphDownSample(nn.Module):
    '''
    Use ST-GCN as encoder, and then use gcn as a decoder
    The input is [batch, in_channels, input_n, node_dim]   # the in_channels is 3 at the beginning
    the out feature of encoder is  [batch, node_dim, feature_len], this is dct feature version.
    '''
    def __init__(self, input_feature, output_feature, samplelist):
        """
        :param input_feature: num of input feature, dct_n
        :param hidden_feature: num of hidden feature
        :param samplelist, a list, every element is the node should be conv together, (16, 17)
        Input is [batch, in_channels, frame_n, node_n], reshape to []
        Output is [batch, out_channels, frame_n, node*_n]
        """
        super(GraphDownSample, self).__init__()

        self.downsample = []
        self.in_channels = input_feature
        self.out_channels = output_feature

        self.list = samplelist
        self.feature = []

        for i in samplelist:
            kernel_size = len(i)
            self.downsample.append(nn.Conv2d(self.in_channels, self.out_channels, (1, kernel_size), (1, 1)))

        self.downsample = nn.ModuleList(self.downsample)

    def forward(self, x):
        for i in range(0, len(self.list)):
            x1 = x[:, :, :, self.list[i]]
            y = self.downsample[i](x1)
            self.feature.append(y)

        y = torch.cat(self.feature, dim=3)
        return y

class GraphUpSample(nn.Module):
    '''
    Use ST-GCN as encoder, and then use gcn as a decoder
    The input is [batch, in_channels, input_n, node_dim]   # the in_channels is 3 at the beginning
    the out feature of encoder is  [batch, node_dim, feature_len], this is dct feature version.
    '''
    def __init__(self, input_feature, output_feature, samplelist):
        """
        :param input_feature: num of input feature, dct_n
        :param hidden_feature: num of hidden feature
        :param samplelist, a list, every element is the node should be conv together, (16, 17)
        Input is [batch, in_channels, frame_n, node_n], reshape to []
        Output is [batch, out_channels, frame_n, node*_n]
        """
        super(GraphUpSample, self).__init__()

        self.upsample = []
        self.in_channels = input_feature
        self.out_channels = output_feature

        self.list = samplelist
        self.feature = []
        self.index = []

        for i in samplelist:
            kernel_size = len(i)
            self.upsample.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, (1, kernel_size), (1, 1)))

        self.upsample = nn.ModuleList(self.upsample)

    def forward(self, x):
        _,_,_, node = x.shape
        l = []
        for i in range(0, node):
            x1 = x[:, :, :, i]
            x1 = torch.unsqueeze(x1, 3)
            y = self.upsample[i](x1)
            self.feature.append(y)
            l.append(self.list[i])

        self.index = list(itertools.chain.from_iterable(l))
        y1 = torch.cat(self.feature, dim=3)
        y = y1
        _, _, _, dims = y.shape
        for i in range(0, dims):
            y[:, :, :, self.index[i]] = y1[:, :, :, i]
        return y
