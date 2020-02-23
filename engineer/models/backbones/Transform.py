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
from torch.nn.modules.transformer import Transformer
from torch.nn.parameter import Parameter


@BACKBONES.register_module
class Transform(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, nhead = 1, num_encoder_layers =100):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(Transform, self).__init__()
        self.trans = Transformer(d_model=66, nhead=nhead, num_encoder_layers=num_encoder_layers)
        self.fcn = nn.Linear(66*10, 66*10)
        self.bn = nn.BatchNorm1d(66*10)
        self.act = nn.LeakyReLU()

    def forward(self, x, padding, targets):
        y = self.trans(x, targets)
        # b = y.size(1)
        # y = y.transpose(1, 0).contiguous().view(-1, 66*10)
        # y = self.bn(y)
        # y = self.act(y)
        # y = self.fcn(y)
        # y = y.contiguous().view(b, 10, 66).transpose(0,1)
        return y