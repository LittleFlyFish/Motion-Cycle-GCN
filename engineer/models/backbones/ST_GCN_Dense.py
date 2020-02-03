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
from engineer.models.backbones.Motion_GCN import Motion_GCN, GraphConvolution, GC_Block


class ST_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, graph_adj, stride=1, dropout=0, residual=True):
        """
        Define a residual block of GCN
        """
        super(ST_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = graph_adj.cuda()

        self.stlayer1 = st_gcn(in_channels, in_channels, kernel_size, 1, residual=False)
        self.stlayer2 = st_gcn(in_channels, out_channels, kernel_size, 1, residual=False)

        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()

    def forward(self, x):
        y, _ = self.stlayer1(x, self.A)
        y = self.act_f(y)
        y = self.do(y)

        y, _ = self.stlayer2(y, self.A)
        y = self.act_f(y)
        y = self.do(y)

        if self.in_channels == self.out_channels:
            y = y + x

        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


@BACKBONES.register_module
class ST_GCN_Dense(nn.Module):
    '''
    Assume the input is a ST-GCN graph, and predict is a out ST-GCN graph.
    The input is [batch, in_channels, input_n, node_dim]   # the in_channels is 3 at the beginning
    the out feature of encoder is  [batch, node_dim, feature_len]
    '''
    def __init__(self, layout, strategy, hidden_feature, dropout, residual, num_stage=1,  **kwargs):
        """

        :param input_feature: num of input feature, dct_n
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(ST_GCN_Dense, self).__init__()

        # load graph
        self.graph = Graph(layout=layout, strategy=strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        self.register_buffer('A', A)
        self.data_bn = nn.BatchNorm1d(3 * A.size(1))

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        input_feature = 3 ## x,y,z as input

        self.num_stage = num_stage
        self.stbs = []
        for i in range(num_stage):
            #self.stbs.append(ST_Block(hidden_feature*(i+1), hidden_feature, kernel_size, graph_adj=A, stride=1, dropout=0, residual=True))
            self.stbs.append(
                ST_Block(hidden_feature, hidden_feature, kernel_size, graph_adj=A, stride=1, dropout=0,
                         residual=True))
        self.stbs = nn.ModuleList(self.stbs)

        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.st1 = st_gcn(input_feature, hidden_feature, kernel_size, 1, residual=False)
        #self.st2 = st_gcn(hidden_feature*(num_stage + 1), input_feature, kernel_size, 1, residual=False)
        self.st2 = st_gcn(hidden_feature, input_feature, kernel_size, 1, residual=False)
        self.residual = residual


    def forward(self, x):
        # # data normalization
        # N, C, T, V = x.size() #[batch, channels, input_n, node_m]
        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = x.view(N, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, V, C, T)
        # x = x.permute(0, 1, 3, 2).contiguous()
        # x = x.view(N, C, T, V)

        # ST-GCN module Dense Version
        y, _ = self.st1(x, self.A)
        y = self.do(y)

        # for i in range(self.num_stage):
        #     y1 = self.stbs[i](y)
        #     y = self.act_f(y)
        #     y = torch.cat((y, y1), dim=1)
        #
        # y, _ = self.st2(y, self.A)
        # if self.residual:
        #     y = y + x

        # ST-GCN module Res Version
        for i in range(self.num_stage):
            y1 = self.stbs[i](y)

        y, _ = self.st2(y, self.A)
        if self.residual:
            y = y + x
        return y