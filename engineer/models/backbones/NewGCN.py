#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
from engineer.models.registry import BACKBONES
from engineer.models.common.Attention import Attention
from engineer.models.backbones.Motion_GCN import GraphConvolution, GC_Block
import numpy as np

class GC_Block_NoRes(nn.Module):
    def __init__(self, in_features, out_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block_NoRes, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, out_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * out_features)

        self.gc3 = GraphConvolution(out_features, out_features, node_n=node_n, bias=bias)
        self.bn3 = nn.BatchNorm1d(node_n * out_features)

        self.gc4 = GraphConvolution(out_features, out_features, node_n=node_n, bias=bias)
        self.bn4 = nn.BatchNorm1d(node_n * out_features)

        self.gc5 = GraphConvolution(out_features, out_features, node_n=node_n, bias=bias)
        self.bn5 = nn.BatchNorm1d(node_n * out_features)

        self.gc6 = GraphConvolution(out_features, out_features, node_n=node_n, bias=bias)
        self.bn6 = nn.BatchNorm1d(node_n * out_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.LeakyReLU()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc3(y)
        b, n, f = y.shape
        y = self.bn3(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc4(y)
        b, n, f = y.shape
        y = self.bn4(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc5(y)
        b, n, f = y.shape
        y = self.bn5(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc6(y)
        b, n, f = y.shape
        y = self.bn6(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


@BACKBONES.register_module
class NewGCN(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, hidden_feature, f_feature, dropout=0.5, input_n=10, output_n=10, node_n=48):
        """
        Input: [batch, node_n, input_n, feature], Output: [batch, node_n, output_n, feature], such as [16, 22, 10, 3]
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param f_feature:
        :param p_dropout: drop out prob.
        :param node_n: number of nodes in graph
        """
        super(NewGCN, self).__init__()

        self.input_n = input_n
        self.output_n = output_n

        self.gcbs = []
        for i in range(self.input_n+self.output_n):
            self.gcbs.append(GC_Block_NoRes(i*hidden_feature + f_feature, hidden_feature, p_dropout=dropout, node_n=node_n))
        self.gcbs = nn.ModuleList(self.gcbs)

        self.decoder = []
        for i in range(self.output_n):
            self.decoder.append(GraphConvolution((i+self.input_n)*hidden_feature, 3, node_n=node_n))
        self.decoder = nn.ModuleList(self.decoder)


        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(node_n * f_feature)
        self.gc1 = GraphConvolution(3, f_feature, node_n=node_n) # output [batch, node, hidden_feature]




    def forward(self, x):
        frame = torch.split(x, 1, dim=2)
        frame = list(frame) # [batch, node, feature, 1]
        g = []
        for i in range(self.input_n):
            f = torch.squeeze(frame[i])
            f1 = self.gc1(f)
            b, n, f_size = f1.shape
            f1 = self.bn1(f1.view(b, -1)).view(b, n, f_size)
            f1 = self.act_f(f1)
            f1 = self.do(f1)
            if i>0:
                g1 = torch.cat((g, f1), dim=2)
            else:
                g1 = f1
            g1 = self.gcbs[i](g1)
            if i>0:
                g = torch.cat((g, g1), dim=2)
            else:
                g = g1


        outputframe =[]
        for i in range(self.output_n):
            outF = self.decoder[i](g) #[batch, node, 3]
            outputframe.append(torch.unsqueeze(outF, dim=2))

            outF1 = self.gc1(outF)
            b, n, f_size = outF1.shape
            outF1 = self.bn1(outF1.view(b, -1)).view(b, n, f_size)
            outF1 = self.act_f(outF1)
            outF1 = self.do(outF1)
            g1 = g
            g1 = torch.cat((g1, outF1), dim=2)
            g1 = self.gcbs[i+self.input_n](g1)
            g = torch.cat((g, g1), dim=2)

        outputframe = torch.cat(outputframe, dim=2)
        return outputframe