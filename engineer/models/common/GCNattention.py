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

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class SpGraphAttentionLayer(nn.Module):
     """
     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
     """

     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
         super(SpGraphAttentionLayer, self).__init__()
         self.in_features = in_features
         self.out_features = out_features
         self.alpha = alpha
         self.concat = concat

         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
         nn.init.xavier_normal_(self.W.data, gain=1.414)

         self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
         nn.init.xavier_normal_(self.a.data, gain=1.414)

         self.dropout = nn.Dropout(dropout)
         self.leakyrelu = nn.LeakyReLU(self.alpha)
         self.special_spmm = SpecialSpmm()

     def forward(self, input, adj):
         dv = 'cuda' if input.is_cuda else 'cpu'

         N = input.size()[0]
         edge = adj.nonzero().t()

         h = torch.mm(input, self.W)
         # h: N x out
         assert not torch.isnan(h).any()

         # Self-attention on the nodes - Shared attention mechanism
         edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
         # edge: 2*D x E

         edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
         assert not torch.isnan(edge_e).any()
         # edge_e: E

         e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
         # e_rowsum: N x 1

         edge_e = self.dropout(edge_e)
         # edge_e: E

         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
         assert not torch.isnan(h_prime).any()
         # h_prime: N x out

         h_prime = h_prime.div(e_rowsum)
         # h_prime: N x out
         assert not torch.isnan(h_prime).any()

         if self.concat:
             # if this layer is not last layer,
             return F.elu(h_prime)
         else:
             # if this layer is last layer,
             return h_prime

     def __repr__(self):
         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    