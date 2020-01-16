import engineer.utils.logging as logging
logger = logging.get_logger(__name__)
from engineer.utils import utils
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd
import os

from engineer.utils import loss_funcs
from engineer.utils import  data_utils as data_utils

def build_dataloader(dataset,num_worker,batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True)

def train_model(model,datasets,cfg,distributed,optimizer):
    # define the leftdim and rightdim
    left = np.array(cfg.left)
    leftdim = np.concatenate((left * 3, left * 3 + 1, left * 3 + 2))
    right = np.array(cfg.right)
    rightdim = np.concatenate((right * 3, right * 3 + 1, right * 3 + 2))

    train_dataset,val_dataset,test_datasets = datasets
    train_loader = build_dataloader(train_dataset,cfg.dataloader.num_worker,cfg.dataloader.batch_size.train)
    val_loader = build_dataloader(val_dataset,cfg.dataloader.num_worker,cfg.dataloader.batch_size.test)
    test_loaders = dict()
    for key in test_datasets.keys():
        test_loaders[key] = build_dataloader(test_datasets[key],cfg.dataloader.num_worker,cfg.dataloader.batch_size.test)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()
    start_epoch = cfg.resume.start
    lr_now = cfg.optim_para.optimizer.lr
    model.train()
    acts = test_loaders.keys()
    test_best =dict()
    for act in acts:
        test_best[act] = float("inf")

    #save_pre_fix
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_3D_in{:d}_out{:d}_dct_n_{:d}'.format(cfg.data.train.input_n, cfg.data.train.output_n, cfg.data.train.dct_used)
    err_best = float("inf")
    is_best_ret_log = None

    for epoch in range(start_epoch, cfg.total_epochs):
        pass
        if (epoch + 1) % cfg.optim_para.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, cfg.optim_para.lr_gamma)


        logger.info('==========================')
        logger.info('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # training on per epoch
        lr_now, t_l = train(train_loader, model, optimizer, lr_now=lr_now, max_norm=cfg.max_norm, is_cuda=is_cuda,
                            dim_used=train_dataset.dim_used, dct_n=cfg.data.train.dct_used,
                            input_n=cfg.data.train.input_n,output_n=cfg.data.train.output_n, rightdim=rightdim, leftdim=leftdim)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        #val evaluation
        v_3d = val(val_loader, model, is_cuda=is_cuda, dim_used=train_dataset.dim_used,
                   dct_n=cfg.data.val.dct_used, rightdim=rightdim, leftdim=leftdim, input_n=cfg.data.train.input_n,output_n=cfg.data.train.output_n)
        ret_log = np.append(ret_log, [v_3d])
        head = np.append(head, ['v_3d'])





        #test_results
        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_l, test_3d = test(test_loaders[act], model, input_n=cfg.data.test.input_n, output_n=cfg.data.test.output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used, dct_n=cfg.data.test.dct_used, rightdim=rightdim, leftdim=leftdim)
            # ret_log = np.append(ret_log, test_l)
            ret_log = np.append(ret_log, test_3d)
            test_best[act] = min(test_best[act],test_3d[0])
            head = np.append(head,
                             [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            if cfg.data.test.output_n > 10:
                head = np.append(head, [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)


        # update log file and save checkpoint
        #output_result
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(cfg.checkpoints + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(cfg.checkpoints + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if not np.isnan(v_3d):
            is_best = v_3d < err_best
            err_best = min(v_3d, err_best)
        else:
            is_best = False
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=cfg.checkpoints,
                        is_best=is_best,
                        file_name=file_name)
        for key in test_best.keys():
            logger.info("{}:{:.4f}".format(key,test_best[key]))

        if is_best:
            is_best_ret_log = ret_log.copy()
    #best ret_log information to save
    df = pd.DataFrame(np.expand_dims(is_best_ret_log, axis=0))
    with open(cfg.checkpoints + '/' + script_name + '.csv', 'a') as f:
        df.to_csv(f, header=False, index=False)

def get_left_input(all_seq, input_n, output_n, dct_n,dim_used, leftdim=[], rightdim=[]):

    all_seq = all_seq[:, :, dim_used] # [batch, framenumber, dim]
    # get input seqs dct
    dct_m_in, _ = data_utils.get_dct_matrix(input_n)
    input_seqs = all_seq[:, 0:input_n, :]
    dct_m_in = Variable(torch.from_numpy(dct_m_in)).float().cuda()
    input_seq_dct = torch.matmul(dct_m_in[0:dct_n, :], input_seqs)
    input_seq_dct = input_seq_dct.transpose(0,1).reshape([-1, len(dim_used), dct_n])

    # get output seqs dct
    dct_m_out, _ = data_utils.get_dct_matrix(output_n)
    output_seqs = all_seq[:, input_n:(input_n+output_n), :]
    dct_m_out = Variable(torch.from_numpy(dct_m_out)).float().cuda()
    output_seq_dct = torch.matmul(dct_m_out[0:dct_n, :], output_seqs)

    output_seq_dct = output_seq_dct.transpose(0,1).reshape([-1, len(dim_used), dct_n])

    # get left and right data for P module
    input_left = input_seq_dct[:, leftdim, :] # batch * leftdim * dct_n
    output_left = output_seq_dct[:, leftdim, :]
    input_right = input_seq_dct[:, rightdim, :]
    output_right = output_seq_dct[:, rightdim, :]




    return input_left, input_seq_dct, output_left, output_seq_dct

def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, dim_used=[], dct_n=15,input_n=10,output_n=10, rightdim=[], leftdim=[]):
    t_l = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()
        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()

        # get the left half input of seq.

        # the model interfer the right side from the left side
        (input_left, input_seq_dct, output_left, output_seq_dct) = \
            get_left_input(all_seq, input_n, output_n, dct_n=dct_n, dim_used=dim_used, leftdim=leftdim, rightdim=rightdim)

        # P_GCN calculate
        P_1_out = model(input_left)
        P_2_out = model(output_left)

        P_inputs = input_seq_dct
        P_inputs[:, rightdim, :] = P_1_out
        P_outputs = output_seq_dct
        P_outputs[:, rightdim, :] = P_2_out

        # calculate loss and backward

        _, loss1 = loss_funcs.mpjpe_error_p3d(P_inputs, all_seq[:,0:input_n,:], dct_n, dim_used)
        _, loss2 = loss_funcs.mpjpe_error_p3d(P_outputs, all_seq[:, input_n:(input_n+output_n), :], dct_n, dim_used)

        loss = loss1 + loss2


        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        t_l.update(loss.item()*batch_size, batch_size)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg
#
#
def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15, rightdim=[], leftdim=[]):
    N = 0
    t_l = 0
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()

        (input_left, input_seq_dct, output_left, output_seq_dct) = \
            get_left_input(all_seq, input_n, output_n, dct_n, dim_used, leftdim=leftdim, rightdim=rightdim)


        # P_GCN calculate
        P_1_out = model(input_left)
        P_2_out = model(output_left)

        P_inputs = input_seq_dct
        P_inputs[:, rightdim, :] = P_1_out
        P_outputs = output_seq_dct
        P_outputs[:, rightdim, :] = P_2_out

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used_len = len(dim_used)

        P1_p3d, loss1 = loss_funcs.mpjpe_error_p3d(P_inputs, all_seq[:,0:input_n,:], dct_n, dim_used)
        P2_p3d, loss2 = loss_funcs.mpjpe_error_p3d(P_outputs, all_seq[:, input_n:(input_n+output_n), :], dct_n, dim_used)

        t_l += loss1 + loss2

        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_l / N, t_3d / N
#
#
def val(train_loader, model, is_cuda=False, dim_used=[], dct_n=15, rightdim=[], leftdim=[], input_n=10, output_n=10):
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()

        (input_left, input_seq_dct, output_left, output_seq_dct) = \
            get_left_input(all_seq, input_n, output_n, dct_n, dim_used, leftdim=leftdim, rightdim=rightdim)


        # P_GCN calculate
        P_1_out = model(input_left)
        P_2_out = model(output_left)

        P_inputs = input_seq_dct
        P_inputs[:, rightdim, :] = P_1_out
        P_outputs = output_seq_dct
        P_outputs[:, rightdim, :] = P_2_out

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used_len = len(dim_used)

        P1_p3d, loss1 = loss_funcs.mpjpe_error_p3d(P_inputs, all_seq[:, 0:input_n, :], dct_n, dim_used)
        P2_p3d, loss2 = loss_funcs.mpjpe_error_p3d(P_outputs, all_seq[:, input_n:(input_n+output_n), :], dct_n, dim_used)

        n, _, _ = all_seq.data.shape

        m_err = loss1 + loss2

        # update the training loss
        t_3d.update(m_err.item() * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d.avg