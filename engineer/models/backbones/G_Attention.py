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



@BACKBONES.register_module
class G_Attention(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, dct_n):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(G_Attention, self).__init__()
        self.attention = Attention(dct_n)


    def forward(self, x):
        y, _ = self.attention(x, x)
        y, _ = self.attention(y, y)
        y, _ = self.attention(y, y)

        return y+x