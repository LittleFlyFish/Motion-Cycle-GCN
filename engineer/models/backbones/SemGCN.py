'''
Semantic Graph Convolutional Networks for 3D Human Pose Regression CVPR 2019

@author Lingteng Qiu
@version : 0.1.0
'''
from __future__ import absolute_import
from engineer.models.registry import BACKBONES
import torch.nn as nn
from engineer.models.common.sem_graph_conv import SemGraphConv
from engineer.models.common.graph_non_local import GraphNonLocal
import torch
import numpy as np
import scipy.sparse as sp

#tools to get adj infomation
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def edge_3d_process(nums,edges):
    '''
    process 3d edge, in here we have 66 points, the serials in here is
    Example
    >>> (x1,y1,z1,x2,y2,z2,...)
    '''
    nums = nums*3
    new_edges = []
    for edge in edges:
        for x in range(edge[0]*3,edge[0]*3+3):
            for y in range(edge[1]*3,edge[1]*3+3):
                new_edges.append([x,y])
    return nums,new_edges
def adj_mx_from_edges(num_pts, edges, sparse=False):
    num_pts,edges = edge_3d_process(num_pts,edges)
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()
        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self._nonlocal = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self._nonlocal(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out

@BACKBONES.register_module
class SemGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN, self).__init__()
        self.adj_matrix = self.get_adj_matrix(adj)

        _gconv_input = [_GraphConv(self.adj_matrix, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []
        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(self.adj_matrix, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            raise NotImplementedError
            # group_size = len(nodes_group[0])
            # assert group_size > 1
            #
            # grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            # restored_order = [0] * len(grouped_order)
            # for i in range(len(restored_order)):
            #     for j in range(len(grouped_order)):
            #         if grouped_order[j] == i:
            #             restored_order[i] = j
            #             break
            #
            # _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            # for i in range(num_layers):
            #     _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
            #     _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], self.adj_matrix)

    @property
    def adj_matrix(self):
        return self._adj
    @adj_matrix.setter
    def adj_matrix(self,adj):
        self._adj =adj
    @staticmethod
    def get_adj_matrix(adj):
        return adj_mx_from_edges(num_pts=adj[0],edges=adj[1])
    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)

        return out+x



