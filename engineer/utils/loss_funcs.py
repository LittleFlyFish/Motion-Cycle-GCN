import numpy as np
import torch
from torch.autograd import Variable
from engineer.utils import data_utils


def sen_loss(outputs, all_seq, dim_used, dct_n):
    """

    :param outputs: N * (seq_len*dim_used_len)
    :param all_seq: N * seq_len * dim_full_len
    :param input_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)
    dim_used = np.array(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    pred_expmap = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    targ_expmap = all_seq.clone()[:, :, dim_used]

    loss = torch.mean(torch.sum(torch.abs(pred_expmap - targ_expmap), dim=2).view(-1))
    return loss


def euler_error(outputs, all_seq, input_n, dim_used, dct_n):
    """

    :param outputs:
    :param all_seq:
    :param input_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp

    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

    # pred_expmap[:, 0:6] = 0
    # targ_expmap[:, 0:6] = 0
    pred_expmap = pred_expmap.view(-1, 3)
    targ_expmap = targ_expmap.view(-1, 3)

    pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors


def mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n):
    """

    :param outputs:
    :param all_seq:
    :param input_n:
    :param dim_used:
    :param data_mean:
    :param data_std:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp
    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len).clone()
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

    pred_expmap[:, 0:6] = 0
    targ_expmap[:, 0:6] = 0

    targ_p3d = data_utils.expmap2xyz_torch(targ_expmap).view(-1, 3)

    pred_p3d = data_utils.expmap2xyz_torch(pred_expmap).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(targ_p3d - pred_p3d, 2, 1))

    return mean_3d_err


def mpjpe_error_cmu(outputs, all_seq, input_n, dim_used, dct_n):
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp
    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)
    pred_expmap[:, 0:6] = 0
    targ_expmap[:, 0:6] = 0

    targ_p3d = data_utils.expmap2xyz_torch_cmu(targ_expmap).view(-1, 3)
    pred_p3d = data_utils.expmap2xyz_torch_cmu(pred_expmap).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(targ_p3d - pred_p3d, 2, 1))

    return mean_3d_err


def mpjpe_error_p3d(outputs, all_seq, dct_n, dim_used):
    """

    :param outputs:n*66*dct_n
    :param all_seq:
    :param dct_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used = np.array(dim_used)
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.contiguous().view(-1, dct_n).transpose(0, 1)
    outputs_p3d = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)
    pred_3d = outputs_p3d.contiguous().view(-1, dim_used_len).view(-1, 3)
    targ_3d = all_seq[:, :, dim_used].contiguous().view(-1, dim_used_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return outputs_p3d, mean_3d_err

def mpjpe_error_p3d_l1(outputs, all_seq, dct_n, dim_used):
    """

    :param outputs:n*66*dct_n
    :param all_seq:
    :param dct_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used = np.array(dim_used)
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.contiguous().view(-1, dct_n).transpose(0, 1)
    outputs_p3d = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)
    pred_3d = outputs_p3d.contiguous().view(-1, dim_used_len).view(-1, 3)
    targ_3d = all_seq[:, :, dim_used].contiguous().view(-1, dim_used_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 1, 1))

    return outputs_p3d, mean_3d_err

def mpjpe_error_p3d_ST(outputs, all_seq, dct_n, dim_used):
    """
    outputs shape: [batch, 3, frame_n, node_n]
    all_seq shape: [batch, frame_n, full_dim]

    :param outputs:n*66*dct_n
    :param all_seq:
    :param dct_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used = np.array(dim_used)
    dim_used_len = len(dim_used)

    outputs_p3d = outputs.contiguous().transpose(1, 2).reshape(n, seq_len, dim_used_len)

    pred_3d = outputs_p3d.contiguous().view(-1, dim_used_len).view(-1, 3)
    targ_3d = all_seq[:, :, dim_used].contiguous().view(-1, dim_used_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return outputs_p3d, mean_3d_err

def R_mpjpe_error_p3d(Gv_Final, allseqs, frame_n, dct_n, dim_used, select_dim):
    """
    this to calculate the MSE(left Input - Final Left predict Input) 1..10 - 1..10 only left

    :param Gv_Final: batch*66*dct_n, the dct of 20 ...1 predict seq
    :param allseqs: the groundtruth inverse seq 1 ..20
    :param frame_n: the input frame number
    :param dct_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = allseqs.data.shape
    dim_used = np.array(dim_used)
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    Gv_Final_t = Gv_Final.view(-1, dct_n).transpose(0, 1)
    Gv_Final_p3d = torch.matmul(idct_m[:, 0:dct_n], Gv_Final_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)
    pred_seq_p3d = torch.flip(Gv_Final_p3d, dims=[1]) # turns to 1...20
    norm_allseqs = allseqs[:, :, dim_used]
    pred_seq_p3d[:, frame_n:seq_len, :] = norm_allseqs[:, frame_n:seq_len, :] # the output error not count
    pred_seq_p3d[:, frame_n:seq_len, select_dim] = norm_allseqs[:, frame_n:seq_len, select_dim] # the select_dim side error not count
    
    
    pred_3d = pred_seq_p3d.contiguous().view(-1, dim_used_len).view(-1, 3)
    targ_3d = allseqs[:, :, dim_used].contiguous().view(-1, dim_used_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return mean_3d_err

def mpjpe_error_3dpw(outputs, all_seq, dct_n, dim_used):
    n, seq_len, dim_full_len = all_seq.data.shape

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_full_len - 3,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)
    pred_3d = all_seq.clone()
    pred_3d[:, :, dim_used] = outputs_exp
    pred_3d = pred_3d.contiguous().view(-1, dim_full_len).view(-1, 3)
    targ_3d = all_seq.contiguous().view(-1, dim_full_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return mean_3d_err
