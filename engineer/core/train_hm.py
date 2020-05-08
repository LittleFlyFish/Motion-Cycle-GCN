import engineer.utils.logging as logging

logger = logging.get_logger(__name__)
from engineer.utils import utils
import time
import random
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
    cuda_num = cfg.cuda_num
    if is_cuda:
        model.cuda(torch.device(cuda_num))
        model.to(cuda_num)
    start_epoch = cfg.resume.start
    lr_now = cfg.optim_para.optimizer.lr

    # ###############################################
    # ## test the checkpoint
    # G_meta = "./checkpoints/_tempconfig/ckpt_train_3D_in10_out10_dct_n_15Original+L1_best.pth.tar"
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
                                                                       cfg.data.train.dct_n) + cfg.flag
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
        lr_now, t_l, t_e, t_3d= train(train_loader, model, optimizer, lr_now=lr_now, input_n=cfg.data.train.input_n,
                                                        max_norm=cfg.max_norm, is_cuda=is_cuda, cuda_num=cuda_num,
                                                        dim_used=train_dataset.dim_used, dct_n=cfg.data.train.dct_n,
                                                        num=train_num, loss_list=train_loss_plot)
        ret_log = np.append(ret_log, [lr_now, t_l, t_e, t_3d])
        head = np.append(head, ['lr', 't_l', 't_e', 't_3d'])

        # val evaluation
        v_e, v_3d = val(val_loader, model, is_cuda=is_cuda, cuda_num=cuda_num, dim_used=train_dataset.dim_used,
                   dct_n=cfg.data.val.dct_n, input_n=cfg.data.val.input_n)
        ret_log = np.append(ret_log, [v_e, v_3d])
        head = np.append(head, ['v_e', 'v_3d'])

        # test_results
        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_e, test_3d = test(test_loaders[act], model, input_n=cfg.data.test.input_n,
                                   output_n=cfg.data.test.output_n, is_cuda=is_cuda, cuda_num=cuda_num,
                                   dim_used=train_dataset.dim_used, dct_n=cfg.data.test.dct_n)
            # ret_log = np.append(ret_log, test_l)
            ret_log = np.append(ret_log, test_e)
            test_best[act] = min(test_best[act], test_3d[0])
            test_3d_temp = np.append(test_3d_temp, test_3d)
            test_3d_head = np.append(test_3d_head,
                                     [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            head = np.append(head, [act + '80', act + '160', act + '320', act + '400'])
            if cfg.data.train.output_n > 10:
                head = np.append(head, [act + '560', act + '1000'])
                test_3d_head = np.append(test_3d_head,
                                         [act + '3d560', act + '3d1000'])
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
        df2.to_csv(f, header=False, index=False)


def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, cuda_num='cuda:0', dim_used=[],
          dct_n=15, num=1, input_n=15,
          loss_list=[1]):
    t_l = utils.AccumLoss()
    t_e = utils.AccumLoss()
    t_3d = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):

        # skip the last batch if only have one sample for batch_norm layers
        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()
        if is_cuda:
            inputs = Variable(inputs.cuda(cuda_num)).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(cuda_num, async=True)).float()

        outputs = model(inputs)
        n = outputs.shape[0]
        outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)

        loss = loss_funcs.sen_loss(outputs, all_seq, dim_used, dct_n, cuda=cuda_num)

        # calculate loss and backward
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        n, _, _ = all_seq.data.shape

        # 3d error
        m_err = loss_funcs.mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n, cuda=cuda_num)

        # angle space error
        e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, dct_n, cuda=cuda_num)

        # update the training loss
        t_l.update(loss.cpu().data.numpy()* n, n)
        t_e.update(e_err.cpu().data.numpy()* n, n)
        t_3d.update(m_err.cpu().data.numpy() * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg, t_e.avg, t_3d.avg

#
def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, cuda_num='cuda:0', dim_used=[], dct_n=15):
    N = 0
    # t_l = 0
    if output_n >= 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]

    t_e = np.zeros(len(eval_frame))
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda(cuda_num)).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(cuda_num, async=True)).float()

        outputs = model(inputs)
        n = outputs.shape[0]
        # outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)

        # loss = loss_funcs.sen_loss(outputs, all_seq, dim_used)

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used_len = len(dim_used)

        # inverse dct transformation
        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda(cuda_num)
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                   seq_len).transpose(1,
                                                                                                                      2)

        pred_expmap = all_seq.clone()
        dim_used = np.array(dim_used)
        pred_expmap[:, :, dim_used] = outputs_exp
        pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
        targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

        pred_expmap[:, 0:6] = 0
        targ_expmap[:, 0:6] = 0
        pred_expmap = pred_expmap.view(-1, 3)
        targ_expmap = targ_expmap.view(-1, 3)

        # get euler angles from expmap
        pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap, cuda=cuda_num), cuda=cuda_num)
        pred_eul = pred_eul.view(-1, dim_full_len).view(-1, output_n, dim_full_len)
        targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap, cuda=cuda_num), cuda=cuda_num)
        targ_eul = targ_eul.view(-1, dim_full_len).view(-1, output_n, dim_full_len)

        # get 3d coordinates
        targ_p3d = data_utils.expmap2xyz_torch(targ_expmap.view(-1, dim_full_len), cuda=cuda_num).view(n, output_n, -1, 3)
        pred_p3d = data_utils.expmap2xyz_torch(pred_expmap.view(-1, dim_full_len), cuda=cuda_num).view(n, output_n, -1, 3)

        # update loss and testing errors
        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_e[k] += torch.mean(torch.norm(pred_eul[:, j, :] - targ_eul[:, j, :], 2, 1)).cpu().data.numpy() * n
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                1)).cpu().data.numpy() * n
        # t_l += loss.cpu().data.numpy()[0] * n
        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_e / N, t_3d / N


#
#
def val(train_loader, model, is_cuda=False, cuda_num='cuda:0', input_n = 15, dim_used=[], dct_n=15):
    # t_l = utils.AccumLoss()
    t_e = utils.AccumLoss()
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda(cuda_num)).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(cuda_num, async=True)).float()

        outputs = model(inputs)
        n = outputs.shape[0]
        outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)

        # loss = loss_funcs.sen_loss(outputs, all_seq, dim_used)

        n, _, _ = all_seq.data.shape
        m_err = loss_funcs.mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n, cuda=cuda_num)
        e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, dct_n, cuda=cuda_num)

        # t_l.update(loss.cpu().data.numpy()[0] * n, n)
        t_e.update(e_err.cpu().data.numpy() * n, n)
        t_3d.update(m_err.cpu().data.numpy() * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_e.avg, t_3d.avg
