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
from engineer.models.common.GraphDownUp import GraphDownSample_Conv, GraphUpSample_Conv
from engineer.models.backbones.Motion_GCN import Motion_GCN, GraphConvolution, GC_Block




@BACKBONES.register_module
class ST_B(nn.Module):
    '''
    Use ST-GCN  do UpSample Downsample autoencoder
    The input is [batch, in_channels, input_n, node_dim]   # the in_channels is 3 at the beginning
    the out feature of encoder is  [batch, node_dim, feature_len], this is dct feature version.
    '''
    def __init__(self,  hidden_feature, layout, strategy, dropout, residual, **kwargs):
        """

        :param input_feature: num of input feature, dct_n
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(ST_B, self).__init__()
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
        in_channels = 3

        spatial_kernel_size = A_d1.size(0)
        temporal_kernel_size = 9
        kernel_size_d1 = (temporal_kernel_size, spatial_kernel_size)



        # self.encoder = nn.ModuleList((
        #     st_gcn(16, 32, kernel_size, 1, residual=False, **kwargs),
        #     st_gcn(32, 64, kernel_size, 2, **kwargs),
        #     st_gcn(64, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 256, kernel_size, 1, **kwargs),
        #     st_gcn(256, 256, kernel_size, 2, **kwargs),
        # ))
        #
        # self.gcn = nn.ModuleList((
        #     GraphConvolution(256*5, 256*3, node_n=22),
        #     GraphConvolution(256 * 3, 256, node_n=22),
        #     GraphConvolution(256, 256, node_n=22),
        #     GraphConvolution(256, 256*3, node_n=22),
        #     GraphConvolution(256*3, 256*5, node_n=22),
        #
        # ))
        #
        # self.decoder = nn.ModuleList((
        #         st_gcn(256, 256, kernel_size, 1, **kwargs),
        #         st_gcn(256, 256, kernel_size, 2, Transpose=True, **kwargs),
        #         st_gcn(256, 256, kernel_size, 2, Transpose=True, **kwargs),
        #         st_gcn(256, 512, kernel_size, 1, **kwargs),
        #         st_gcn(512, 512, kernel_size, 1, **kwargs),
        #     ))

        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.st1 = st_gcn(in_channels, hidden_feature, kernel_size, 1, residual=False)
        self.st2 = st_gcn(hidden_feature, hidden_feature, kernel_size_d1, 1, residual=False)
        self.st3 = st_gcn(hidden_feature, hidden_feature, kernel_size, 1, residual=False)
        self.st4 = st_gcn(hidden_feature, in_channels, kernel_size, 1, residual=False)
        self.residual = residual

        list1 = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15,16], [17,18,19,20,21]]
        self.gd1 = GraphDownSample_Conv(hidden_feature, hidden_feature, list1)
        self.gu1 = GraphUpSample_Conv(hidden_feature, hidden_feature, list1)

        list2 = [[0,1,2,3,4]]
        self.gd2 = GraphDownSample_Conv(hidden_feature, hidden_feature, list2)
        self.gu2= GraphUpSample_Conv(hidden_feature, hidden_feature, list2)

    def forward(self, x):
        y, _ = self.st1(x, self.A)
        y = self.act_f(y)
        y = self.do(y)
        batch, feature, frame_n, node = y.shape

        y = self.gd1(y)
        y = self.act_f(y)
        y = self.do(y)

        y, _ = self.st2(y, self.A_d1)
        y = self.act_f(y)
        y = self.do(y)
        u1 = y

        y = self.gd2(y)
        y = self.act_f(y)
        y = self.do(y)

        u2 = y

        y = self.gu2(y)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gu1(y+u1)
        y = self.act_f(y)
        y = self.do(y)

        y, _ = self.st3(y, self.A)
        y = self.act_f(y)
        y = self.do(y)

        y, _ = self.st4(y, self.A)
        y = self.act_f(y)
        y = self.do(y)

        return y