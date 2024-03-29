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
class ST_D(nn.Module):
    '''
    Use GCN as encoder, and then use gcn as a decoder
    The input is [batch, node_dim, dct_n]   # for example, [16, 66, 15]
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
        super(ST_D, self).__init__()
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

        spatial_kernel_size = A_d1.size(0)
        temporal_kernel_size = 9
        kernel_size_d1 = (temporal_kernel_size, spatial_kernel_size)

        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.gc1 = GraphConvolution(in_channels, hidden_feature, node_n=66)
        self.gc2 = GraphConvolution(hidden_feature, hidden_feature, node_n=15)
        self.gc3 = GraphConvolution(hidden_feature, hidden_feature, node_n=66)
        self.gc4 = GraphConvolution(hidden_feature, in_channels, node_n=66)
        self.residual = residual

        list1 = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15,16], [17,18,19,20,21]]
        self.gd1 = GraphDownSample(hidden_feature, hidden_feature, list1)
        self.gu1 = GraphUpSample(hidden_feature, hidden_feature, list1)

        list2 = [[0,1,2,3,4]]
        self.gd2 = GraphDownSample(hidden_feature, hidden_feature, list2)
        self.gu2= GraphUpSample(hidden_feature, hidden_feature, list2)

    def forward(self, x):
        y = self.gc1(x) #[16, 66, 256]
        print(y.shape)
        y = self.act_f(y)
        y = self.do(y)
        batch, n, f = y.shape
        y = y.transpose(1,2).reshape(batch, f, 3, 22)

        y = self.gd1(y)
        print(y.shape) #[16, 256, 3, 5]
        y = self.act_f(y)
        y = self.do(y)
        y = y.view(batch, -1, 3*5).transpose(1,2)

        y = self.gc2(y)
        print(y.shape) #[16,15,256]
        y = self.act_f(y)
        y = self.do(y)
        y = y.transpose(1,2).reshape(batch, self.hidden_feature, 3, 5)
        u1 = y

        y = self.gd2(y)#[16, 256, 3, 1]
        y = self.act_f(y)
        y = self.do(y)
        print(y.shape)

        u2 = y

        y = self.gu2(y)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gu1(y)
        y = self.act_f(y)
        y = self.do(y)
        y = y.view(batch, -1, 66).transpose(1,2)

        y = self.gc3(y)
        y = self.act_f(y)
        y = self.do(y)

        y= self.gc4(y)


        return y + x