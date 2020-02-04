#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import itertools

class GraphDownSample(nn.Module):
    def __init__(self, input_feature, output_feature, samplelist):
        """
        :param input_feature must equals to output features.
        :param samplelist, a list, every element is the node should be conv together, (16, 17)
        GCN Input [batch, node*3, feature]
        ST_GCN Input [Batch, feature, frame_n, node_n]
        for this module:
        Input is [batch, feature, in_channel,  node_n], the start in_channel is 3, will change after ST_GCN layer
        Output is [batch, feature, in_channel,  node*_n]
        """
        super(GraphDownSample, self).__init__()

        self.downsample = []
        self.in_channels = input_feature
        self.out_channels = output_feature

        self.list = samplelist
        self.feature = []

        for i in samplelist:
            kernel_size = len(i)
            #self.downsample.append(nn.Conv2d(self.in_channels, self.out_channels, (1, kernel_size), (1, 1)))
            self.downsample.append(nn.Linear(kernel_size*3, 3))


        self.downsample = nn.ModuleList(self.downsample)

    def forward(self, x):
        self.feature = []
        batch, feature, _, node_n = x.shape
        for i in range(0, len(self.list)):
            x1 = x[:, :, :, self.list[i]]
            x1 = x1.contiguous().view(batch*feature, -1)
            y = self.downsample[i](x1)
            y = y.reshape(batch, feature, 3, 1)
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
            #self.upsample.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, (1, kernel_size), (1, 1)))
            self.upsample.append(nn.Linear(3, 3*kernel_size))

        self.upsample = nn.ModuleList(self.upsample)

    def forward(self, x):
        _,_,_, node = x.shape
        l = []
        batch, features, _, node_n = x.shape # [16, 256, 3, 22]
        self.feature = []
        for i in range(0, node):
            x1 = x[:, :, :, i] # [batch, feature, 3, 1]
            x1 = x1.contiguous().view(batch*features, -1)
            y = self.upsample[i](x1)
            y = y.reshape(batch*features, 3, len(self.list[i])).reshape(batch, features, 3, len(self.list[i]))
            self.feature.append(y)
            l.append(self.list[i])

        self.index = list(itertools.chain.from_iterable(l))
        y1 = torch.cat(self.feature, dim=3)
        y = y1
        _, _, _, dims = y.shape
        for i in range(0, dims):
            y[:, :, :, self.index[i]] = y1[:, :, :, i]
        return y

class GraphDownSample_Avg(nn.Module):
    def __init__(self, input_feature, output_feature, samplelist):
        """
        :param input_feature must equals to output features.
        :param samplelist, a list, every element is the node should be conv together, (16, 17)
        GCN Input [batch, node*3, feature]
        ST_GCN Input [Batch, feature, frame_n, node_n]
        for this module:
        Input is [batch, feature, in_channel,  node_n], the start in_channel is 3, will change after ST_GCN layer
        Output is [batch, feature, in_channel,  node*_n]
        """
        super(GraphDownSample_Avg, self).__init__()

        self.downsample = []
        self.in_channels = input_feature
        self.out_channels = output_feature

        self.list = samplelist
        self.feature = []

        for i in samplelist:
            kernel_size = len(i)
            #self.downsample.append(nn.Conv2d(self.in_channels, self.out_channels, (1, kernel_size), (1, 1)))
            self.downsample.append(nn.Linear(kernel_size*3, 3))


        self.downsample = nn.ModuleList(self.downsample)

    def forward(self, x):
        self.feature = []
        batch, feature, _, node_n = x.shape
        for i in range(0, len(self.list)):
            x1 = x[:, :, :, self.list[i]]
            y = x1.sum(3)
            y = torch.unsqueeze(dim=3)

            self.feature.append(y)

        y = torch.cat(self.feature, dim=3)
        return y

class GraphUpSample_Avg(nn.Module):
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
        super(GraphUpSample_Avg, self).__init__()

        self.upsample = []
        self.in_channels = input_feature
        self.out_channels = output_feature

        self.list = samplelist
        self.feature = []
        self.index = []

        for i in samplelist:
            kernel_size = len(i)
            #self.upsample.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, (1, kernel_size), (1, 1)))
            self.upsample.append(nn.Linear(3, 3*kernel_size))

        self.upsample = nn.ModuleList(self.upsample)

    def forward(self, x):
        _,_,_, node = x.shape
        l = []
        batch, features, _, node_n = x.shape # [16, 256, 3, 22]
        self.feature = []
        for i in range(0, node):
            x1 = x[:, :, :, i] # [batch, feature, 3, 1]
            y = x1.repeat(1,1,1,len(self.list[i]))
            self.feature.append(y)
            l.append(self.list[i])

        self.index = list(itertools.chain.from_iterable(l))
        y1 = torch.cat(self.feature, dim=3)
        y = y1
        _, _, _, dims = y.shape
        for i in range(0, dims):
            y[:, :, :, self.index[i]] = y1[:, :, :, i]
        return y

class GraphDownSample_Conv(nn.Module):
    def __init__(self, input_feature, output_feature, samplelist):
        """
        :param input_feature must equals to output features.
        :param samplelist, a list, every element is the node should be conv together, (16, 17)
        GCN Input [batch, node*3, feature]
        ST_GCN Input [Batch, feature, frame_n, node_n]
        for this module:
        Input is [batch, feature, in_channel,  node_n], the start in_channel is 3, will change after ST_GCN layer
        Output is [batch, feature, in_channel,  node*_n]
        """
        super(GraphDownSample_Conv, self).__init__()

        self.downsample = []
        self.in_channels = input_feature
        self.out_channels = output_feature

        self.list = samplelist
        self.feature = []

        for i in samplelist:
            kernel_size = len(i)
            self.downsample.append(nn.Conv2d(self.in_channels, self.in_channels, (1, kernel_size), (1, 1)))
            self.downsample.append(nn.Conv2d(self.in_channels, self.out_channels, (1,1), (1,1)))

        self.downsample = nn.ModuleList(self.downsample)

    def forward(self, x):
        self.feature = []
        batch, feature, _, node_n = x.shape
        for i in range(0, len(self.list)):
            x1 = x[:, :, :, self.list[i]]
            y = self.downsample[2*i](x1)
            y = self.downsample[2*i+1](y)
            self.feature.append(y)

        y = torch.cat(self.feature, dim=3)
        return y

class GraphUpSample_Conv(nn.Module):
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
        super(GraphUpSample_Conv, self).__init__()

        self.upsample = []
        self.in_channels = input_feature
        self.out_channels = output_feature

        self.list = samplelist
        self.feature = []
        self.index = []

        for i in samplelist:
            kernel_size = len(i)
            self.upsample.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, (1, kernel_size), (1, 1)))
            self.upsample.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, (1, 1), (1, 1)))

        self.upsample = nn.ModuleList(self.upsample)

    def forward(self, x):
        _,_,_, node = x.shape
        l = []
        batch, features, _, node_n = x.shape # [16, 256, 3, 22]
        self.feature = []
        for i in range(0, node):
            x1 = x[:, :, :, i] # [batch, feature, 3, 1]
            x1 = torch.unsqueeze(x1, dim=3)
            y = self.upsample[2*i](x1)
            y = self.upsample[2*i+1](y)
            self.feature.append(y)
            l.append(self.list[i])

        self.index = list(itertools.chain.from_iterable(l))
        y1 = torch.cat(self.feature, dim=3)
        y = y1
        _, _, _, dims = y.shape
        for i in range(0, dims):
            y[:, :, :, self.index[i]] = y1[:, :, :, i]
        return y

class GraphDownSample_Pool(nn.Module):
    def __init__(self, input_feature, output_feature, samplelist):
        """
        :param input_feature must equals to output features.
        :param samplelist, a list, every element is the node should be conv together, (16, 17)
        GCN Input [batch, node*3, feature]
        ST_GCN Input [Batch, feature, frame_n, node_n]
        for this module:
        Input is [batch, feature, in_channel,  node_n], the start in_channel is 3, will change after ST_GCN layer
        Output is [batch, feature, in_channel,  node*_n]
        """
        super(GraphDownSample_Pool, self).__init__()

        self.downsample = []
        self.in_channels = input_feature
        self.out_channels = output_feature

        self.list = samplelist
        self.feature = []

        for i in samplelist:
            kernel_size = len(i)
            self.downsample.append(nn.MaxPool2d(self.in_channels, self.in_channels, (1, kernel_size), (1, 1)))

        self.downsample = nn.ModuleList(self.downsample)

    def forward(self, x):
        self.feature = []
        batch, feature, _, node_n = x.shape
        for i in range(0, len(self.list)):
            x1 = x[:, :, :, self.list[i]]
            y = self.downsample[i](x1)
            self.feature.append(y)

        y = torch.cat(self.feature, dim=3)
        return y
