#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from engineer.models.registry import BACKBONES
from engineer.models.backbones.Motion_GCN import GraphConvolution, GC_Block
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from engineer.models.common.tgcn import ConvTemporalGraphical
from engineer.models.common.graph import Graph
from engineer.models.common.STGCN import st_gcn
from engineer.models.common.GraphDownUp import GraphDownSample, GraphUpSample
from engineer.models.backbones.Motion_GCN import Motion_GCN, GraphConvolution, GC_Block


@BACKBONES.register_module
class Subnet_GCN(nn.Module):
    '''
    Use GCN as encoder, and then use gcn as a decoder
    The input is [batch, node_dim, dct_n]   # for example, [16, 66, 15]
    the out feature of encoder is  [batch, node_dim, feature_len], this is dct feature version.
    '''

    def __init__(self, input_feature, hidden_feature, dropout, num_stage=1, node_n=48, residual=True):
        """

        :param input_feature: num of input feature, dct_n
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(Subnet_GCN, self).__init__()

        in_channels = input_feature
        self.hidden_feature = hidden_feature
        self.in_channels = in_channels

        self.do = nn.Dropout(dropout)
        self.act =  nn.LeakyReLU()
        self.gc1 = GraphConvolution(in_channels, hidden_feature, node_n=66)
        self.gc1l = GraphConvolution(in_channels, hidden_feature, node_n=39)
        self.gc1r = GraphConvolution(in_channels, hidden_feature, node_n=39)
        self.gc1v1 = GraphConvolution(in_channels, hidden_feature, node_n=7*3)
        self.gc1v2 = GraphConvolution(in_channels, hidden_feature, node_n=6*3)
        self.gc1v3 = GraphConvolution(in_channels, hidden_feature, node_n=7*3)
        self.gc1v4 = GraphConvolution(in_channels, hidden_feature, node_n=6*3)
        self.gc7 = GraphConvolution(3 * hidden_feature, in_channels, node_n=66)
        self.gc7l = GraphConvolution(hidden_feature, in_channels,  node_n=39)
        self.gc7r = GraphConvolution(hidden_feature, in_channels, node_n=39)
        self.gc7v1 = GraphConvolution(hidden_feature, in_channels, node_n=7*3)
        self.gc7v2 = GraphConvolution(hidden_feature, in_channels, node_n=6*3)
        self.gc7v3 = GraphConvolution(hidden_feature, in_channels, node_n=7*3)
        self.gc7v4 = GraphConvolution(hidden_feature, in_channels, node_n=6*3)

        self.residual = residual
        node_n = 66

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=node_n))
        self.gcbs = nn.ModuleList(self.gcbs)

        self.gcbsl = []
        for i in range(num_stage):
            self.gcbsl.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=39))

        self.gcbsl = nn.ModuleList(self.gcbsl)

        self.gcbsr = []
        for i in range(num_stage):
            self.gcbsr.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=39))
        self.gcbsr = nn.ModuleList(self.gcbsr)

        self.gcbv1 = []
        for i in range(num_stage):
            self.gcbv1.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=7*3))
        self.gcbv1 = nn.ModuleList(self.gcbv1)

        self.gcbv2 = []
        for i in range(num_stage):
            self.gcbv2.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=6*3))
        self.gcbv2 = nn.ModuleList(self.gcbv2)

        self.gcbv3 = []
        for i in range(num_stage):
            self.gcbv3.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=7*3))
        self.gcbv3 = nn.ModuleList(self.gcbv3)

        self.gcbv4 = []
        for i in range(num_stage):
            self.gcbv4.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=6*3))
        self.gcbv4 = nn.ModuleList(self.gcbv4)


        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)  # 15 is in_channel
        self.bn1l = nn.BatchNorm1d(39 * hidden_feature)  # 15 is in_channel
        self.bn1r = nn.BatchNorm1d(39 * hidden_feature)  # 15 is in_channel
        self.bn1v1 = nn.BatchNorm1d(7*3 * hidden_feature)  # 15 is in_channel
        self.bn1v2 = nn.BatchNorm1d(6*3 * hidden_feature)  # 15 is in_channel
        self.bn1v3 = nn.BatchNorm1d(7*3 * hidden_feature)  # 15 is in_channel
        self.bn1v4 = nn.BatchNorm1d(6*3 * hidden_feature)  # 15 is in_channel
        self.num_stage = num_stage

        left = np.array([0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # the index of left parts of INPUT data
        self.leftdim = np.concatenate((left * 3, left * 3 + 1, left * 3 + 2))
        right = np.array([4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21])  # the index of the right parts of the INPUT data
        self.rightdim = np.concatenate((right * 3, right * 3 + 1, right * 3 + 2))

        V1 = np.array([0, 1, 2, 3, 8, 9, 10]) # 7
        V2 = np.array([11, 12, 13, 14, 15, 16]) # 6
        V3 = np.array([4, 5, 6, 7, 8, 9, 10])  # 7
        V4 = np.array([11, 17, 18, 19, 20, 21]) # 6
        self.V1dim = np.concatenate((V1 * 3, V1 * 3 + 1, V1 * 3 + 2))
        self.V2dim = np.concatenate((V2 * 3, V2 * 3 + 1, V2 * 3 + 2))
        self.V3dim = np.concatenate((V3 * 3, V3 * 3 + 1, V3 * 3 + 2))
        self.V4dim = np.concatenate((V4 * 3, V4 * 3 + 1, V4 * 3 + 2))


    def forward(self, x):

        x_left = x[:, self.leftdim, :]
        x_right = x[:, self.rightdim, :]
        x_V1 = x[:, self.V1dim, :]
        x_V2 = x[:, self.V2dim, :]
        x_V3 = x[:, self.V3dim, :]
        x_V4 = x[:, self.V4dim, :]

        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        yl = self.gc1l(x_left)
        b, n, f = yl.shape
        yl = self.bn1l(yl.view(b, -1)).view(b, n, f)
        yl = self.act(yl)
        yl = self.do(yl)
        for i in range(self.num_stage):
            yl = self.gcbsl[i](yl)

        yr = self.gc1r(x_right)
        b, n, f = yr.shape
        yr = self.bn1r(yr.view(b, -1)).view(b, n, f)
        yr = self.act(yr)
        yr = self.do(yr)
        for i in range(self.num_stage):
            yr = self.gcbsr[i](yr)

        yv1 = self.gc1v1(x_V1)
        b, n, f = yv1.shape
        yv1 = self.bn1v1(yv1.view(b, -1)).view(b, n, f)
        yv1 = self.act(yv1)
        yv1 = self.do(yv1)
        for i in range(self.num_stage):
            yv1 = self.gcbv1[i](yv1)

        yv2 = self.gc1v2(x_V2)
        b, n, f = yv2.shape
        yv2 = self.bn1v2(yv2.view(b, -1)).view(b, n, f)
        yv2 = self.act(yv2)
        yv2 = self.do(yv2)
        for i in range(self.num_stage):
            yv2 = self.gcbv2[i](yv2)

        yv3 = self.gc1v3(x_V3)
        b, n, f = yv3.shape
        yv3 = self.bn1v3(yv3.view(b, -1)).view(b, n, f)
        yv3 = self.act(yv3)
        yv3 = self.do(yv3)
        for i in range(self.num_stage):
            yv3 = self.gcbv3[i](yv3)

        yv4 = self.gc1v4(x_V4)
        b, n, f = yv4.shape
        yv4 = self.bn1v4(yv4.view(b, -1)).view(b, n, f)
        yv4 = self.act(yv4)
        yv4 = self.do(yv4)
        for i in range(self.num_stage):
            yv4 = self.gcbv4[i](yv4)

        ytotal= y.clone()
        yV = y.clone()
        ytotal[:, self.leftdim, :] = yl
        ytotal[:, self.rightdim, :] = yr
        yV[:, self.V1dim, :] = yv1
        yV[:, self.V2dim, :] = yv2
        yV[:, self.V3dim, :] = yv3
        yV[:, self.V4dim, :] = yv4

        y = torch.cat([yV, ytotal, y], dim=2)
        # y = torch.cat([ytotal, y], dim=2)

        if self.residual == True:
            y = self.gc7(y)
            y = y + x
            yr = self.gc7r(yr)
            yr = yr + x_right
            yl = self.gc7l(yl)
            yl = yl + x_left

        return y, yl, yr