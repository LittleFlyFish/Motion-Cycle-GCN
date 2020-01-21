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
from engineer.models.backbones.Motion_GCN import Motion_GCN



@BACKBONES.register_module
class G_Attention_GCN(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48, dct_n=15):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(G_Attention_GCN, self).__init__()
        self.attention = Attention(dct_n)
        self.gcn = Motion_GCN(input_feature, hidden_feature, p_dropout, num_stage, node_n)


    def forward(self, x):
        y = self.gcn(x)
        y, _ = self.attention(y, x)
        return y+x