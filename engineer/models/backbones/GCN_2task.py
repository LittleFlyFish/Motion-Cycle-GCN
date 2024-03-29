#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from engineer.models.registry import BACKBONES
from engineer.models.common.Attention import Attention


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

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

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

@BACKBONES.register_module
class GCN_2task(nn.Module):
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
        super(GCN_2task, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        self.bn2 = nn.BatchNorm1d(node_n * input_feature)
        self.bn3 = nn.BatchNorm1d(node_n * input_feature)
        self.bn7 = nn.BatchNorm1d(node_n * input_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.gc8 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.gc9 = GraphConvolution(input_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.act_f1 = nn.LeakyReLU()
        self.residual = residual
        self.att = Attention(15)

        self.W = nn.Parameter(torch.randn(4))

        self.fcn = nn.Linear(15*66, 15)
        self.Soft = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        if self.residual == True:
            e1 = self.gc7(y)
            e2 = self.gc8(y)

            y1, _ = self.att(e1, e1)
            y2, _ = self.att(e2, e2)

            # y1 = (self.W[0]*e1 + self.W[1]*e2)/(self.W[0] + self.W[1])
            # y2 = (self.W[2]*e1 + self.W[3]*e2)/(self.W[2] + self.W[3])

            # context = torch.cat([torch.unsqueeze(e1.view(-1, 66*15), dim=1), torch.unsqueeze(e2.view(-1, 66*15), dim=1)], dim=1)
            # ee1, _ = self.att(torch.unsqueeze(e1.view(-1, 66*15), dim=1), context)
            # ee2, _ = self.att(torch.unsqueeze(e2.view(-1, 66*15), dim=1), context)

            # context = torch.cat([e1.transpose(1,2), e2.transpose(1,2)], dim=1)
            # ee1, _ = self.att(e1.transpose(1,2), context)
            # ee2, _ = self.att(e2.transpose(1,2), context)
            #
            # y1 = ee1.transpose(1,2)
            # y2 = ee2.transpose(1,2) # .view(b, 66, 15)

            b, n, f = y1.shape
            y1 = self.bn3(y1.contiguous().view(b, -1)).contiguous().view(b, n, f)
            y1 = self.act_f(y1)
            y1 = self.do(y1)
            y1 = self.gc9(y1)

            y1 = y1 + x
            b, n, f = y2.shape
            y2 = self.bn2(y2.contiguous().view(b, -1)).contiguous().view(b, n, f)
            y2 = self.act_f(y2)
            y2 = self.do(y2)
            y2 = y2.view(-1, 15*66)
            y2 = self.fcn(y2)
            y2 = self.act_f1(y2)
            y2 = self.Soft(y2)


        #else:
            # y = self.gc7(y)
            # b, n, f = y.shape
            # y = self.bn7(y.view(b, -1)).view(b, n, f)
            # y = self.act_f(y)
            # y = self.do(y)

        return y1, y2