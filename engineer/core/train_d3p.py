#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd

from engineer.utils import loss_funcs
from engineer.utils import  data_utils as data_utils
from engineer.utils import utils


def build_dataloader(dataset,num_worker,batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True)

def train_model(model,datasets,cfg,optimizer):

    train_dataset,val_dataset,test_dataset = datasets
    train_loader = build_dataloader(train_dataset,cfg.dataloader.num_worker, cfg.dataloader.batch_size.train)
    val_loader = build_dataloader(val_dataset,cfg.dataloader.num_worker,cfg.dataloader.batch_size.test)
    test_loader = build_dataloader(test_dataset, cfg.dataloader.num_worker, cfg.dataloader.batch_size.test)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()
        model.to('cuda:0')
    start_epoch = cfg.resume.start
    lr_now = cfg.optim_para.optimizer.lr

    model.train()

    #save_pre_fix
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_3D_in{:d}_out{:d}_dct_n_{:d}'.format(cfg.data.train.input_n, cfg.data.train.output_n, cfg.data.train.dct_n)
    err_best = float("inf")
    is_best_ret_log = None
    train_num = 0

    for epoch in range(0, cfg.total_epochs):

        if (epoch + 1) % cfg.optim_para.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, cfg.optim_para.lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch

        # lr_now, t_l, train_num, train_loss_plot = train(train_loader, model, optimizer, lr_now=lr_now, max_norm=cfg.max_norm, is_cuda=is_cuda,
        #                     dim_used=train_dataset.dim_used, dct_n=cfg.data.train.dct_used, num=train_num, loss_list=train_loss_plot)

        lr_now, t_l, t_err = train(train_loader, model,
                             optimizer,
                             lr_now=lr_now,
                             max_norm=cfg.max_norm,
                             is_cuda=is_cuda,
                             dct_n=cfg.data.train.dct_n,
                             dim_used=train_dataset.dim_used,
                            cuda=cfg.cuda_num)

        ret_log = np.append(ret_log, [lr_now, t_l, t_err])
        head = np.append(head, ['lr', 't_l', 't_err'])

        v_err = val(val_loader, model,
                   is_cuda=is_cuda,
                   dct_n=cfg.data.val.dct_n,
                   dim_used=val_dataset.dim_used,
                   cuda = cfg.cuda_num)

        ret_log = np.append(ret_log, v_err)
        head = np.append(head, ['v_err'])

        test_3d = test(test_loader, model,
                       input_n=cfg.data.test.input_n,
                       output_n=cfg.data.test.output_n,
                       is_cuda=is_cuda,
                       dim_used=test_dataset.dim_used,
                       dct_n=cfg.data.test.dct_n,
                       cuda=cfg.cuda_num
                       )

        ret_log = np.append(ret_log, test_3d)
        if output_n == 15:
            head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d'])
        elif output_n == 30:
            head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d', '6003d', '7003d', '8003d', '9003d',
                                    '10003d'])

        # update log file
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(cfg.checkpoints + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(cfg.checkpoints + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        # save ckpt
        is_best = v_3d < err_best
        err_best = min(v_3d, err_best)
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=cfg.checkpoints,
                        is_best=is_best,
                        file_name=file_name)


def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, dct_n=15, dim_used=[], cuda='cuda:0'):
    sen_l = utils.AccumLoss()
    eul_err = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):

        batch_size = inputs.shape[0]
        # batch size is 1 do not train
        if batch_size == 1:
            break

        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda(cuda, )).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(cuda, async=True)).float()
        else:
            inputs = Variable(inputs).float()
            # targets = Variable(targets).float()
            all_seq = Variable(all_seq).float()
        outputs = model(inputs)
        loss = loss_funcs.sen_loss(outputs, all_seq, dim_used, dct_n, cuda=cuda)

        # calculate loss and backward
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        n, seq_len, _ = all_seq.data.shape
        # update the training loss
        e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, dct_n, cuda=cuda)
        sen_l.update(loss.cpu().data.numpy()[0] * n * seq_len, n * seq_len)
        eul_err.update(e_err.cpu().data.numpy()[0] * n * seq_len, n * seq_len)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, sen_l.avg, eul_err.avg


def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15, cuda='cuda:0'):
    N = 0
    if output_n == 15:
        eval_frame = [2, 5, 8, 11, 14]
    elif output_n == 30:
        eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    t_err = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda(cuda)).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(cuda, async=True)).float()
        else:
            inputs = Variable(inputs).float()
            # targets = Variable(targets).float()
            all_seq = Variable(all_seq).float()
        outputs = model(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape

        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda(cuda)
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_full_len - 3,
                                                                                                   seq_len).transpose(1,
                                                                                                                      2)
        pred_exp = all_seq.clone()
        pred_exp[:, :, dim_used] = outputs_exp
        pred_exp = pred_exp.contiguous().view(n, seq_len, -1)[:, input_n:, :]
        targ_exp = all_seq.contiguous().view(n, seq_len, -1)[:, input_n:, :]

        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_err[k] += torch.mean(torch.norm(targ_exp[:, j, :] - pred_exp[:, j, :], p=2, dim=1)).cpu().data.numpy() * n

        # update the training loss
        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_err / N


def val(train_loader, model, is_cuda=False, dim_used=[], dct_n=15, cuda='cuda:0'):
    t_err = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda(cuda)).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(cuda, async=True)).float()
        else:
            inputs = Variable(inputs).float()
            # targets = Variable(targets).float()
            all_seq = Variable(all_seq).float()
        outputs = model(inputs)
        e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, dct_n, cuda=cuda)

        n, seq_len, _ = all_seq.data.shape
        # update the training loss
        t_err.update(e_err.cpu().data.numpy() * n * seq_len, n * seq_len)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_err.avg


