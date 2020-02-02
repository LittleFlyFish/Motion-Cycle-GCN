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
from engineer.models.backbones.NewGCN import GC_Block_NoRes
import numpy as np

class GCNGRU_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, node_n=48, dtype=float):
        """
        Define a residual block of GCN
        """
        super(GCNGRU_Block, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_n = node_n

        self.gc1 = GraphConvolution(input_dim + hidden_dim, 2*self.hidden_dim, node_n=node_n, bias=bias)
        self.gc2 = GraphConvolution(input_dim + hidden_dim, self.hidden_dim, node_n=node_n, bias=bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.node_n, self.hidden_dim)).type(self.dtype))


    def forward(self, h, x):
        if x is None:
            combine = h
        else:
            combine = torch.cat([h, x], dim=2)
        combine_gcn = self.gc1(combine)

        gamma, beta = torch.split(combine_gcn, self.hidden_dim, dim=2)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combine = torch.cat([x, reset_gate*h], dim=2)
        cc_cnm = self.gc2(combine)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h + update_gate * cnm

        return h_next

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

@BACKBONES.register_module
class GCNGRU(nn.Module):
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
        super(GCNGRU, self).__init__()

        self.input_n= input_n
        self.output_n = output_n

        self.gcbs = []
        for i in range(self.input_n+self.output_n):
            self.gcbs.append(GC_Block_NoRes(f_feature, hidden_feature, p_dropout=0.5, bias=True, node_n=node_n))
        self.gcbs = nn.ModuleList(self.gcbs)

        self.decoder = []
        for i in range(self.output_n):
            self.decoder.append(GraphConvolution(hidden_feature, 3, node_n=node_n))
        self.decoder = nn.ModuleList(self.decoder)

        self.do = nn.Dropout(dropout)
        self.act_f = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(node_n * f_feature)
        self.gc1 = GraphConvolution(3, f_feature, node_n=node_n) # output [batch, node, hidden_feature]
        self.attention = Attention(f_feature)



    def forward(self, x):
        frame = torch.split(x, 1, dim=2)
        frame = list(frame) # [batch, node, feature, 1]
        for i in range(self.input_n):
            f = torch.squeeze(frame[i])
            f1 = self.gc1(f)
            b, n, f_size = f1.shape
            f1 = self.bn1(f1.view(b, -1)).view(b, n, f_size)
            f1 = self.act_f(f1)
            f1 = self.do(f1)
            if i==0:
                g1 = self.gcbs[i](f1, f1)
            else:
                g1 = self.gcbs[i](g1, f1)



        outputframe =[]
        for i in range(self.output_n):
            outF = self.decoder[i](g1) #[batch, node, 3]
            outputframe.append(torch.unsqueeze(outF, dim=2))

            outF1 = self.gc1(outF)
            b, n, f_size = outF1.shape
            outF1 = self.bn1(outF1.view(b, -1)).view(b, n, f_size)
            outF1 = self.act_f(outF1)
            outF1 = self.do(outF1)

            g1, _ = self.attention(g1, outF1)
            g1 = self.gcbs[i+self.input_n](g1, outF1)

        outputframe = torch.cat(outputframe, dim=2)
        return outputframe

class GCNGRU2(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(GCNGRU2, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(GCNGRU_Block(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t=frame_n, node, c) or (t=frame_n,b,n,c) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :], # (b,t,n,c)
                                              h_cur=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param