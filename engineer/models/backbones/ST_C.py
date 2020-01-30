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
class ST_C(nn.Module):
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
        super(ST_C, self).__init__()
        # load graph
        self.graph = Graph(layout=layout, strategy=strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        in_channels = 3


        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.in_feature = 512
        self.st2 = st_gcn(self.in_feature, in_channels, kernel_size, 1, residual=False)
        self.st1 = st_gcn(in_channels, self.in_feature, kernel_size, 1, residual=False)
        self.gcn = Motion_GCN(self.in_feature*10, 256, 0.5, num_stage=12, node_n=22)

        self.residual = residual


    def forward(self, x):

        y, _ = self.st1(x, self.A) # [16, 16, 10, 22]
        batch, feature, frame_n, node_n = y.shape
        y = y.reshape(batch, feature*frame_n, node_n).transpose(1, 2)
        y = self.gcn(y)
        y = y.transpose(1, 2).reshape(batch, feature, frame_n, node_n)
        y, _ = self.st2(y, self.A)
        return y