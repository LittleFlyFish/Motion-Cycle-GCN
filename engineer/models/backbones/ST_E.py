#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function


from engineer.models.registry import BACKBONES

import torch
import torch.nn as nn
from engineer.models.common.graph import Graph
from engineer.models.common.STGCN import st_gcn
from engineer.models.common.GraphDownUp import GraphDownSample_Conv, GraphUpSample_Conv
from engineer.models.backbones.Motion_GCN import Motion_GCN, GraphConvolution, GC_Block

class STGCN_encoder(nn.Module):
    def __init__(self, hidden_feature, layout, strategy, p_dropout, bias=True, node_n=48):
        """
        Use a ST_GCN as encoder
        input: [batch, in_channels, input_n, node_dim]  such as [16, 3, 10, 22]
        output: [batch, 2*hidden_feature, input_n/2/2, 1] such as [16, 2*h, 2, 1]
        """
        super(STGCN_encoder, self).__init__()
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

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.LeakyReLU()
        self.st1 = st_gcn(in_channels, hidden_feature, kernel_size, 1, residual=False)
        self.st2 = st_gcn(hidden_feature, hidden_feature, kernel_size_d1, 1, residual=False)
        self.st3 = st_gcn(hidden_feature, hidden_feature, kernel_size, 2, residual=False)
        self.st4 = st_gcn(hidden_feature, hidden_feature, kernel_size, 2, residual=False)

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
        y, _ = self.st3(y, self.A) # temporal downsample
        y = self.act_f(y)
        y = self.do(y)

        y, _ = self.st4(y, self.A) # temporal downsample
        y = self.act_f(y)
        y = self.do(y)


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
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


@BACKBONES.register_module
class ST_E(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, hidden_feature, layout="h36m", strategy="spatial", dropout=0.5, input_n=10, output_n=10, node_n=48):
        """
        Input: [batch, node_n, input_n, feature], Output: [batch, node_n, output_n, feature], such as [16, 22, 10, 3]
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param f_feature:
        :param p_dropout: drop out prob.
        :param node_n: number of nodes in graph
        """
        super(ST_E, self).__init__()

        self.input_n = input_n
        self.output_n = output_n

        self.longencoder = STGCN_encoder(hidden_feature, layout, strategy, p_dropout=dropout)
        self.shortencoder = STGCN_encoder(hidden_feature, layout, strategy, p_dropout=dropout)
        self.decoder = nn.Linear(hidden_feature*5, 66)

        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        self.gc1 = GraphConvolution(3, hidden_feature, node_n=node_n) # output [batch, node, hidden_feature]

    def forward(self, x):
        batch, _, _, _ = x.shape
        frames = torch.split(x, 1, dim=2)
        frames = list(frames)
        longfeature = self.longencoder(x)
        for i in range(self.output_n):
            shortlist = frames[(i+self.input_n - 5) : (i+self.input_n)]
            shortx = torch.cat(shortlist, dim=2)
            shortfeature = self.shortencoder(shortx)
            DF = torch.cat([shortfeature, longfeature], dim=2)
            DF = DF.view(batch, -1)
            OutFrame = self.decoder(DF)
            OutFrame = OutFrame.reshape([batch, 3, 1, 22])
            OutFrame = OutFrame + frames[self.input_n-1]
            frames.append(OutFrame)

        outputframe = frames[self.input_n: self.input_n + self.output_n]
        outputframe = torch.cat(outputframe, dim=2)
        return outputframe