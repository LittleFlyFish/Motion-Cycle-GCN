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
from engineer.utils import data_utils as data_utils


# plotter = data_utils.VisdomLinePlotter(env_name='Train Plots')

def build_dataloader(dataset, num_worker, batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True)


def train_model(model, datasets, cfg, distributed, optimizer):
    train_dataset, val_dataset, test_datasets = datasets
    train_loader = build_dataloader(train_dataset, cfg.dataloader.num_worker, cfg.dataloader.batch_size.train)
    val_loader = build_dataloader(val_dataset, cfg.dataloader.num_worker, cfg.dataloader.batch_size.test)
    test_loaders = dict()
    for key in test_datasets.keys():
        test_loaders[key] = build_dataloader(test_datasets[key], cfg.dataloader.num_worker,
                                             cfg.dataloader.batch_size.test)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()
        model.to('cuda:0')
    start_epoch = cfg.resume.start
    lr_now = cfg.optim_para.optimizer.lr

    # ###############################################
    # ## test the checkpoint
    # G_meta = "./checkpoints/Motion_GCN_I10_O10_D15_G/ckpt_train_3D_in10_out10_dct_n_15_best.pth.tar"
    # model.load_state_dict(torch.load(G_meta)["state_dict"])
    # ###############################################

    model.train()
    acts = test_loaders.keys()
    test_best = dict()
    for act in acts:
        test_best[act] = float("inf")

    # save_pre_fix
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_3D_in{:d}_out{:d}_dct_n_{:d}'.format(cfg.data.train.input_n, cfg.data.train.output_n,
                                                                       cfg.data.train.dct_used)
    err_best = float("inf")
    is_best_ret_log = None
    train_num = 0
    train_loss_plot = [1]
    test_loss_plot = [1]
    val_loss_plot = [1]

    for epoch in range(start_epoch, cfg.total_epochs):
        pass
        if (epoch + 1) % cfg.optim_para.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, cfg.optim_para.lr_gamma)

        logger.info('==========================')
        logger.info('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # training on per epoch
        lr_now, t_l, train_num, train_loss_plot = train(train_loader, model, optimizer, lr_now=lr_now,
                                                        max_norm=cfg.max_norm, is_cuda=is_cuda,
                                                        dim_used=train_dataset.dim_used, dct_n=cfg.data.train.dct_used,
                                                        num=train_num, loss_list=train_loss_plot)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        # val evaluation
        v_3d = val(val_loader, model, is_cuda=is_cuda, dim_used=train_dataset.dim_used, dct_n=cfg.data.val.dct_used)
        ret_log = np.append(ret_log, [v_3d])
        head = np.append(head, ['v_3d'])

        # test_results
        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_l, test_3d = test(test_loaders[act], model, input_n=cfg.data.test.input_n,
                                   output_n=cfg.data.test.output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used, dct_n=cfg.data.test.dct_used)
            # ret_log = np.append(ret_log, test_l)
            ret_log = np.append(ret_log, test_3d)
            test_best[act] = min(test_best[act], test_3d[0])
            head = np.append(head,
                             [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            if cfg.data.test.output_n > 10:
                head = np.append(head, [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)

        # update log file and save checkpoint
        # output_result
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
            logger.info("{}:{:.4f}".format(key, test_best[key]))

        if is_best:
            is_best_ret_log = ret_log.copy()
    # best ret_log information to save
    df = pd.DataFrame(np.expand_dims(is_best_ret_log, axis=0))
    with open(cfg.checkpoints + '/' + script_name + '.csv', 'a') as f:
        df.to_csv(f, header=False, index=False)

    df2 = pd.DataFrame(data={"train_loss": train_loss_plot})
    with open(cfg.checkpoints + '/' + script_name + '_loss.csv', 'a') as f:
        df.to_csv(f, header=False, index=False)


def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, dim_used=[], dct_n=15, num=1,
          loss_list=[1]):
    t_l = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, padding_seq, all_seq) in enumerate(train_loader):
        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()
            padding_seq = Variable(padding_seq.cuda()).float()

        outputs = model(inputs)
        Final = torch.cat([inputs, outputs+padding_seq], 2)
        # calculate loss and backward
        _, loss = loss_funcs.mpjpe_error_p3d_ST(Final, all_seq, dct_n, dim_used)
        # #############################################################################
        # #debug loss
        #
        # a_all_seqs = all_seq[:, :, dim_used]
        # reshape_seq = a_all_seqs.reshape(16, 20, 3, 22).transpose(1, 2)
        # print(reshape_seq.shape)
        # _, loss = loss_funcs.mpjpe_error_p3d_ST(inputs[:, :, :, :], all_seq[:, :, :], dct_n, dim_used)
        # print(loss)
        # #############################################################################

        num += 1
        # plotter.plot('loss', 'train', 'LeakyRelu+No Batch ', num, loss.item())
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        t_l.update(loss.item() * batch_size, batch_size)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg, num, loss_list


#
#
def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15):
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
    for i, (inputs, padding_seq, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()
            padding_seq = Variable(padding_seq.cuda()).float()

        outputs = model(inputs)
        dim_used_len = len(dim_used)

        n, seq_len, dim_full_len = all_seq.data.shape

        Final = torch.cat([inputs, outputs + padding_seq], 2)
        outputs_3d = Final.contiguous().transpose(1, 2).reshape(n, seq_len, dim_used_len)
        # calculate loss and backward
        _, test_loss = loss_funcs.mpjpe_error_p3d_ST(Final, all_seq, dct_n, dim_used)
        # plotter.plot('loss', 'test', 'LeakyRelu+No Batch ', i, test_loss.item())

        pred_3d = all_seq.clone()
        dim_used = np.array(dim_used)


        # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        pred_3d[:, :, dim_used] = outputs_3d
        pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
        pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
        targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]

        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                1)).item() * n

        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_l / N, t_3d / N


#
#
def val(train_loader, model, is_cuda=False, dim_used=[], dct_n=15):
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, padding_seq, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()
            padding_seq = Variable(padding_seq.cuda()).float()

        outputs = model(inputs)

        n, _, _ = all_seq.data.shape

        Final = torch.cat([inputs, outputs + padding_seq], dim=2)
        # calculate loss and backward
        _, m_err = loss_funcs.mpjpe_error_p3d_ST(Final, all_seq, dct_n, dim_used)
        # plotter.plot('loss', 'val', 'LeakyRelu+No Batch ', i, m_err.item())

        # update the training loss
        t_3d.update(m_err.item() * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d.avg
