 #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from engineer.models.registry import BACKBONES
import numpy as np
from engineer.models.common.GCNattention import SpGAT, SpGraphAttentionLayer, GAT, GraphAttentionLayer
from engineer.models.backbones.Motion_GCN import GC_Block, GraphConvolution
from engineer.models.common.graph import Graph

@BACKBONES.register_module
class ST_GCNA(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, input_feature, hidden_feature, p_dropout, layout, strategy, num_stage=1, node_n=48, residual=True):
        """
        input = [batch, node, dct_n]
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(ST_GCNA, self).__init__()
        # load graph
        self.graph = Graph(layout=layout, strategy=strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)


        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        self.bn7 = nn.BatchNorm1d(node_n * input_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.ga1 = GraphAttentionLayer(hidden_feature, hidden_feature, dropout=0.5, alpha=0.2, concat=True)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.residual = residual

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        print(torch.squeeze(self.A, 0))
        print(torch.squeeze(self.A, 0).size())
        y2 = self.ga1(y, torch.squeeze(self.A, 0))
        print(y2.shape)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        if self.residual == True:
            y = self.gc7(y)
            y = y + x
        #else:
            # y = self.gc7(y)
            # b, n, f = y.shape
            # y = self.bn7(y.view(b, -1)).view(b, n, f)
            # y = self.act_f(y)
            # y = self.do(y)

        return y