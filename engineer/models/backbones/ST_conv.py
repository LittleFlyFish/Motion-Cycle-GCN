#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from engineer.models.registry import BACKBONES
from engineer.models.backbones.Motion_GCN import Motion_GCN, GraphConvolution, GC_Block



@BACKBONES.register_module
class ST_conv(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48, residual=True):
        """
        input = [batch, node, dct_n]
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(ST_conv, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature + 16 + 14 + 11, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        self.bn7 = nn.BatchNorm1d(node_n * input_feature)
        self.bnc1 = nn.BatchNorm1d(node_n * 16)
        self.bnc2 = nn.BatchNorm1d(node_n * 14)
        self.bnc3 = nn.BatchNorm1d(node_n * 11)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.conv1 = nn.Conv1d(node_n, node_n, 5, stride=1)
        self.conv2 = nn.Conv1d(node_n, node_n, 7, stride=1)
        self.conv3 = nn.Conv1d(node_n, node_n, 10, stride=1)
        self.Iconv1 = nn.ConvTranspose1d(node_n, node_n, 5, stride=1)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.residual = residual

    def forward(self, x, seq): # x = [16, 66, 15], seq = [16, 20, 66], output = [16, 66, 15]
        y_c1 = self.conv1(seq.transpose(1, 2)) # [16, 66, 16]
        b, n, f = y_c1.shape
        y_c1 = self.bnc(y_c1.view(b, -1)).view(b, n, f)
        y_c1 = self.act_f(y_c1)
        y_c1 = self.do(y_c1)

        y_c2 = self.conv2(seq.transpose(1, 2)) # [16, 66, 14]
        b, n, f = y_c2.shape
        y_c2 = self.bnc1(y_c2.view(b, -1)).view(b, n, f)
        y_c2 = self.act_f(y_c2)
        y_c2 = self.do(y_c2)

        y_c3 = self.conv2(seq.transpose(1, 2)) # [16, 66, 11]
        b, n, f = y_c3.shape
        y_c3 = self.bnc1(y_c3.view(b, -1)).view(b, n, f)
        y_c3 = self.act_f(y_c3)
        y_c3 = self.do(y_c3)

        y = torch.cat((y_c1, y_c2, y_c3, x), dim=2)
        # y = y_c

        y = self.gc1(y)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        if self.residual == True:
            y = self.gc7(y)
            # b, n, f = y.shape
            # y = self.bn7(y.view(b, -1)).view(b, n, f)
            # y = self.act_f(y)
            # y = self.do(y)
            # y = self.Iconv1(y)
            y1 = y + x

        #else:
            # y = self.gc7(y)
            # b, n, f = y.shape
            # y = self.bn7(y.view(b, -1)).view(b, n, f)
            # y = self.act_f(y)
            # y = self.do(y)

        return y1