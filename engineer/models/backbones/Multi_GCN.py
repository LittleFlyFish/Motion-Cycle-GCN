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
class Multi_GCN(nn.Module):
    '''
    Use GCN as encoder, and then use gcn as a decoder
    The input is [batch, node_dim, dct_n]   # for example, [16, 66, 15]
    the out feature of encoder is  [batch, node_dim, feature_len], this is dct feature version.
    '''
    def __init__(self,  hidden_feature, layout, strategy, dropout, num_stage, residual, **kwargs):
        """

        :param input_feature: num of input feature, dct_n
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(Multi_GCN, self).__init__()
        # load graph
        self.graph = Graph(layout=layout, strategy=strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.graph_d1 = Graph(layout="h36m_d1", strategy=strategy)
        A_d1 = torch.tensor(self.graph_d1.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_d1', A_d1)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        in_channels = 15
        self.hidden_feature = hidden_feature
        self.in_channels = in_channels

        spatial_kernel_size = A_d1.size(0)
        temporal_kernel_size = 9
        kernel_size_d1 = (temporal_kernel_size, spatial_kernel_size)

        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.gc1 = GraphConvolution(in_channels, hidden_feature, node_n=66)
        self.gc2 = GraphConvolution(hidden_feature, in_channels, node_n=66)
        self.gc3 = GraphConvolution(hidden_feature + in_channels, in_channels, node_n=66)
        self.gc4 = GraphConvolution(hidden_feature + 2 * in_channels, in_channels, node_n=66)

        self.residual = residual
        node_n = 66

        self.gcbs = []
        for i in range(num_stage[0] + num_stage[1] + num_stage[2]):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)


        # #list1 = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15,16], [17,18,19,20,21]]
        # list1 = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7], [8,9], [9,10], [10,11],
        #            [12,13], [13,14], [15], [14,16], [17,18], [18,19], [20], [19,21]] #17
        #
        # list2 = [[0,1,2], [3,4,5], [6,7,8], [9,10,11,12], [13,14,15,16]]
        # list3 = [[0,1,2,3,4]]
        # self.gd1 = GraphDownSample(in_channels, in_channels, list1)
        # self.gu1 = GraphUpSample(in_channels + hidden_feature, in_channels + hidden_feature, list1)
        #
        # self.gd2 = GraphDownSample(hidden_feature, hidden_feature, list2)
        # self.gu2= GraphUpSample(hidden_feature, hidden_feature, list2)
        #
        # self.gcn = Motion_GCN(input_feature=in_channels + hidden_feature, hidden_feature= hidden_feature, p_dropout=0.5, num_stage=12, node_n=17*3, residual=False)
        #
        # self.fullgcn = Motion_GCN(input_feature=in_channels, hidden_feature=hidden_feature, p_dropout=0.5, num_stage=12, node_n=66, residual=False)

        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature) # 15 is in_channel
        self.bn2 = nn.BatchNorm1d(node_n * in_channels) # 15 is in_channel
        self.bn3 = nn.BatchNorm1d(node_n * in_channels) # 15 is in_channel
        self.num_stage = num_stage


    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage[0]):
            y = self.gcbs[i](y)

        y1a = self.gc2(y)
        b, n, f = y1a.shape
        y1 = self.bn2(y1a.view(b, -1)).view(b, n, f)
        y1 = self.act_f(y1)
        y1 = self.do(y1)

        for i in range(self.num_stage[1]):
            y = self.gcbs[i + self.num_stage[0]](y)

        y2a = self.gc3(torch.cat([y1, y], dim=2))
        b, n, f = y2a.shape
        y2 = self.bn3(y2a.view(b, -1)).view(b, n, f)
        y2 = self.act_f(y2)
        y2 = self.do(y2)

        for i in range(self.num_stage[2]):
            y = self.gcbs[i + self.num_stage[0] + self.num_stage[1]](y)

        y3a = self.gc4(torch.cat([y2, y], dim=2))

        y1a = y1a + x
        y2a = y2a + x
        y3a = y3a + x
        return y1a, y2a, y3a