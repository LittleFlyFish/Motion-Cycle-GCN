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
        self.act = nn.Tanh() # nn.LeakyReLU()
        self.gc1 = GraphConvolution(in_channels, hidden_feature, node_n=66)
        self.gc1l = GraphConvolution(in_channels, hidden_feature, node_n=33)
        self.gc1r = GraphConvolution(in_channels, hidden_feature, node_n=33)
        self.gc7 = GraphConvolution(2 * hidden_feature, in_channels, node_n=66)

        self.residual = residual
        node_n = 66

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gcbsl = []
        for i in range(num_stage):
            self.gcbsl.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=33))

        self.gcbsl = nn.ModuleList(self.gcbsl)

        self.gcbsr = []
        for i in range(num_stage):
            self.gcbsr.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=33))

        self.gcbsr = nn.ModuleList(self.gcbsr)

        # #list1 = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15,16], [17,18,19,20,21]]
        # list1 = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7], [8,9], [9,10], [10,11],
        #            [12,13], [13,14], [15], [14,16], [17,18], [18,19], [20], [19,21]] #17
        #
        # list2 = [[0,1,2], [3,4,5], [6,7,8], [9,10,11,12], [13,14,15,16]]
        # list3 = [[0,1,2,3,4]]

        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)  # 15 is in_channel
        self.bn1l = nn.BatchNorm1d(33 * hidden_feature)  # 15 is in_channel
        self.bn1r = nn.BatchNorm1d(33 * hidden_feature)  # 15 is in_channel
        self.num_stage = num_stage

    def forward(self, x):

        x_left = x[:, 0:33, :]
        x_right = x[:, 33:66, :]
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
        yr = self.bn1(yr.view(b, -1)).view(b, n, f)
        yr = self.act(yr)
        yr = self.do(yr)

        for i in range(self.num_stage):
            yr = self.gcbsr[i](yr)

        ytotal[:, 0:33, :] = yl
        ytotal[:, 33:66, :] = yr

        y = torch.cat([ytotal, y], dim=2)

        if self.residual == True:
            y = self.gc7(y)
            y = y + x

        return y