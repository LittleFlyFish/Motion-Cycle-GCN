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



@BACKBONES.register_module
class Transform(nn.Module):
    '''
    Original Module GCN structure
    '''
    def __init__(self, nhead = 1, num_encoder_layers =10):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(Transform, self).__init__()
        self.trans = Transformer(d_model=66, nhead = nhead, num_encoder_layers =num_encoder_layers)


    def forward(self, x, padding, targets):
        y = self.trans(x, targets)
        return y + padding