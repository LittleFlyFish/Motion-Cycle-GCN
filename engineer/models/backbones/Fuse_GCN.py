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
class Fuse_GCN(nn.Module):
    '''
    Use ST-GCN as encoder, and then use gcn as a decoder
    The input is [batch, in_channels, input_n, node_dim]   # the in_channels is 3 at the beginning
    the out feature of encoder is  [batch, node_dim, feature_len], this is dct feature version.
    '''
    def __init__(self, graph_args,  input_n, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48, **kwargs):
        """

        :param input_feature: num of input feature, dct_n
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        in_channels = 3


        super(Fuse_GCN, self).__init__()
        self.num_stage = num_stage
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.motion_gcn = Motion_GCN(hidden_feature, hidden_feature, p_dropout, num_stage, node_n)
        self.gc1 = GraphConvolution(256*input_n, hidden_feature, node_n=node_n)
        self.gc2 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)


    def forward(self, x, x_dct):
        batch, in_channels, input_n, node_dim = x.shape
        for st_gcn in zip(self.st_gcn_networks):  # pass through the ST-GCN to encoder the feature
            y, _ = st_gcn(x)
        y = y.contiguous.view(batch, -1, node_dim).transpose(1, 2)
        y = self.gc1(y) # encoder feature: [batch, node_dim, hidden_feature]
        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x_dct

        return y