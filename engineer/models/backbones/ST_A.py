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





@BACKBONES.register_module
class ST_A(nn.Module):
    '''
    Use ST-GCN as encoder, and then use gcn as a decoder
    The input is [batch, in_channels, input_n, node_dim]   # the in_channels is 3 at the beginning
    the out feature of encoder is  [batch, node_dim, feature_len], this is dct feature version.
    '''
    def __init__(self,  layout, strategy, dropout, residual, **kwargs):
        """

        :param input_feature: num of input feature, dct_n
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(ST_A, self).__init__()
        # load graph
        self.graph = Graph(layout=layout, strategy=strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        in_channels = 3



        self.st_gcn_networks = nn.ModuleList((
            st_gcn(32, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.st2 = st_gcn(256, in_channels, kernel_size, 1, residual=False)
        self.st1 = st_gcn(in_channels, 32, kernel_size, 1, residual=False)
        self.residual = residual

    def forward(self, x):
        y, _ = self.st1(x, self.A)

        for st_gcn in self.st_gcn_networks:  # pass through the ST-GCN to encoder the feature
            y, _ = st_gcn(y, self.A)

        y = self.act_f(y)
        y = self.do(y)
        y, _ = self.st2(y, self.A)
        if self.residual:
            y = y + x

        return y