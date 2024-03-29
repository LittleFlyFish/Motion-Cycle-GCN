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
import torch.nn.functional as F

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
        # input = [batch, node, in_f], adj = [node, node]
        h = torch.matmul(input, self.W) #[batch, node, out_f]
        b = h.size()[0]
        N = h.size()[1] #[node]

        a_input = torch.cat([h.repeat(1, 1, N).view(b, N * N, -1), h.repeat(1, N, 1)], dim=2).view(b, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) #[16, 22, 22]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training) #[16, 22, 22]
        h_p = torch.matmul(h.permute(0, 2, 1), attention)
        h_prime = h_p.permute(0, 2, 1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[1]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


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

     def forward(self, inputBatch, adj):
         dv = 'cuda' if inputBatch.is_cuda else 'cpu'
         b = inputBatch.size()[0]
         inputList = torch.split(inputBatch, 1, dim=0)
         t = 0
         ResultList = [None]*b
         for input in inputList:
             input = input.squeeze(0) # [22, 45]
             N = input.size()[0]  # 22
             edge = adj.nonzero().t()  # non zero edge [2， 52]
             h = torch.mm(input, self.W)  # [22, 3*256]
             h[h != h] = 0
             # h: N x out
             assert not torch.isnan(h).any()

             # Self-attention on the nodes - Shared attention mechanism
             edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
             # edge: 2*D x E

             edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
             edge_e[edge_e != edge_e] = 0
             assert not torch.isnan(edge_e).any()
             # edge_e: E

             O_e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
             # e_rowsum: N x 1

             nonzeros = torch.nonzero(O_e_rowsum)
             e_rowsum = torch.ones(size=(N, 1), device=dv)
             e_rowsum[nonzeros[:, 0]] = O_e_rowsum[nonzeros[:, 0]]

             edge_e = self.dropout(edge_e)
             # edge_e: E

             h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)  # [22, 3*256]
             h_prime[h_prime != h_prime] = 0
             assert not torch.isnan(h_prime).any()
             # h_prime: N x out

             h_prime = h_prime.div(e_rowsum)
             h_prime[h_prime != h_prime] = 0
             # h_prime: N x out
             assert not torch.isnan(h_prime).any()

             if self.concat:
                 # if this layer is not last layer,
                 results = F.elu(h_prime)
             else:
                 # if this layer is last layer,
                 results = h_prime # [22, 3 * 256]
             ResultList[t] = results.unsqueeze(0)
             t = t + 1
         results = torch.cat(ResultList, dim=0)
         return results



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
        """Input: [node, nfeat], output: [node, nclass]"""
        x = F.dropout(x, self.dropout, training=self.training) # [batch, node, nfeat]
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2) # [batch, node, nhid * nhead]
        x = F.dropout(x, self.dropout, training=self.training) # [batch, node, nhid * nhead]
        x = F.elu(self.out_att(x, adj)) # [node, nclass]
        return x


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


