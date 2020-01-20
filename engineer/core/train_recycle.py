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

plotter = data_utils.VisdomLinePlotter(env_name='Recycle Plots')

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
    p_dct = cfg.p_dct
    train_num = 0


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
        lr_now, t_l, train_num = train(train_loader, model, optimizer, lr_now=lr_now, max_norm=cfg.max_norm, is_cuda=is_cuda,
                            dim_used=train_dataset.dim_used, dct_n=cfg.data.train.dct_used, p_dct = p_dct,
                            input_n=cfg.data.train.input_n,output_n=cfg.data.train.output_n,
                            rightdim=rightdim, leftdim=leftdim, num= train_num)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        #val evaluation
        v_3d = val(val_loader, model, is_cuda=is_cuda, dim_used=train_dataset.dim_used,
                   dct_n=cfg.data.val.dct_used, rightdim=[], leftdim=[])
        ret_log = np.append(ret_log, [v_3d])
        head = np.append(head, ['v_3d'])





        #test_results
        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_l, test_3d = test(test_loaders[act], model, input_n=cfg.data.test.input_n, output_n=cfg.data.test.output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used, dct_n=cfg.data.test.dct_used, rightdim=[], leftdim=[])
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

def get_reverse_input(g_out_3d,input_n,output_n,dct_used,dim_used):
    #reverse output from G
    all_seqs = torch.flip(g_out_3d,dims=[1])

    dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
    pad_idx = np.repeat([output_n - 1], input_n)
    i_idx = torch.from_numpy(np.append(np.arange(0, output_n), pad_idx))

    pad_all_seqs =all_seqs[:,i_idx,:]
    input_dct_seq = data_utils.seq2dct(pad_all_seqs, dct_n=dct_used)

    return input_dct_seq

def get_left_input(all_seq, input_n, output_n, dct_n,dim_used, leftdim=[], rightdim=[]):

    all_seq = all_seq[:, :, dim_used] # [batch, framenumber, dim]
    # get input seqs dct
    dct_m_in, _ = data_utils.get_dct_matrix(input_n)
    input_seqs = all_seq[:, 0:input_n, :]
    input_seq_dct = data_utils.seq2dct(input_seqs, dct_n)

    # get output seqs dct
    dct_m_out, _ = data_utils.get_dct_matrix(output_n)
    output_seqs = all_seq[:, input_n:(input_n+output_n), :]
    output_seq_dct = data_utils.seq2dct(output_seqs, dct_n)

    # get left and right data for P module
    input_left = input_seq_dct[:, leftdim, :] # batch * leftdim * dct_n
    output_left = output_seq_dct[:, leftdim, :]
    input_right = input_seq_dct[:, rightdim, :]
    output_right = output_seq_dct[:, rightdim, :]

    return input_left, input_right, input_seq_dct, output_left, output_right, output_seq_dct

def Short2Long(P_Input, input_n, output_n, dct_n):
    # this function turns back dct of short Input(1..10) to dct of padding long seq 1..10,10,10,..10
    Short_seq = data_utils.dct2seq(P_Input, input_n)
    idx = np.append(np.arange(0, input_n), np.repeat([input_n - 1], output_n))
    Long_seq = Short_seq[:, idx, :]
    Long_dct = data_utils.seq2dct(Long_seq, dct_n)
    return Long_dct

def Long2Short(G_Output, input_n, output_n, dct_n, dim=[]):
    # this function turns back dct of 1 .. 20 to the dct of right side of 10..20
    Long_seq = data_utils.dct2seq(G_Output, input_n+output_n)
    Short_seq = Long_seq[:, input_n:(input_n+output_n), :]
    Short_dct = data_utils.seq2dct(Short_seq, dct_n)
    Pv_O_right = Short_dct[:, dim, :]
    return Pv_O_right

def VShort2Long(Pv_Output, input_n, output_n, dct_n):
    # change 10..20 of P* to 20 ..10..10..10 input feed in G*
    Vshort_seq = data_utils.dct2seq(Pv_Output, output_n)
    Vshort_seq = torch.flip(Vshort_seq, dims=[1]) # 20 ...10
    idx = np.append(np.arange(0, output_n), np.repeat([output_n - 1], input_n))
    Long_seq = Vshort_seq[:, idx, :] # 20 ...10, 10, ..10
    dct_Long = data_utils.seq2dct(Long_seq, dct_n)
    return dct_Long
    



def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, dim_used=[],
          dct_n=15, p_dct = 5, input_n=10, output_n=10, rightdim=[], leftdim=[], num=1):
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


        # Recycle 1: G*P*GP(left) = left
        # the model infer the right side from the left side
        (input_left, input_right, input_seq_dct, output_left, output_right, output_seq_dct) = \
            get_left_input(all_seq, input_n, output_n, dct_n=p_dct, dim_used=dim_used, leftdim=leftdim,rightdim=rightdim)
        P_I_right = model.p(input_left) # input_left = [batch, leftnode, dct_n]
        P_Input = input_seq_dct
        P_Input[:, rightdim, :] = P_I_right # generate the input for G, the dct of 1..10(input) frame
        G_Input = Short2Long(P_Input, input_n, output_n, dct_n=dct_n) # generate the dct of 1...20(input+padding) frame
        G_Output = model.g(G_Input) # generate the dct of 1..20 (input+output)
        Pv_O_right = Long2Short(G_Output, input_n, output_n, p_dct, rightdim) # generate the dct of 10..20
        Pv_O_left = model.p_verse(Pv_O_right)
        Pv_Output = output_seq_dct
        Pv_Output[:, leftdim, :] = Pv_O_left # generate the inputs for G*
        Gv_Input= VShort2Long(Pv_Output, input_n, output_n, dct_n) # generate the dct of 20 ..1 (output + padding)
        Gv_Output = model.g_verse(Gv_Input) # the predict dct of 20..1
        loss1R = loss_funcs.R_mpjpe_error_p3d(Gv_Output, all_seq, input_n, dct_n, dim_used, rightdim)
            # this loss only calculate the 10..1 left side error
        
        # Recycle 2: GPG*P*(right) = right
        # the model infer the left side from the right side of output
        Pv_O_left = model.p_verse(output_right)
        Pv_Out = output_seq_dct
        Pv_Out[:, leftdim, :] = Pv_O_left
        Gv_In = VShort2Long(Pv_Out, input_n, output_n, dct_n)
        Gv_Out = model.g_verse(Gv_In)
        P_In = Long2Short(Gv_Out, input_n, output_n, p_dct, leftdim) # obtain left dct feature for P
        P_Out = model.p(P_In)
        G_In = Short2Long(P_Out, input_n, output_n, dct_n)
        print(G_In.shape)
        G_Out = model.g(G_In)
        loss2L = loss_funcs.R_mpjpe_error_p3d(G_Out, torch.flip(all_seq, dims=1), output_n, dct_n, dim_used, leftdim)
        

        # Cycle Constrains: GG* = I, G*G = I

        # P*P(left) = left calculate
        P_I_right = model.p(input_left)
        P_O_right = model.p(output_left)

        PvP_I_left = model.p_verse(P_I_right)
        PvP_O_left = model.p_verse(P_O_right)

        PvP_inputs = input_seq_dct
        PvP_inputs[:, leftdim, :] = PvP_I_left
        PvP_outputs = output_seq_dct
        PvP_outputs[:, leftdim, :] = PvP_O_left

        # calculate the left loss and backward

        _, loss1 = loss_funcs.mpjpe_error_p3d(PvP_inputs, all_seq[:, 0:input_n, :], dct_n, dim_used)
        _, loss2 = loss_funcs.mpjpe_error_p3d(PvP_outputs, all_seq[:, input_n:(input_n + output_n), :], dct_n, dim_used)
        loss_left = loss1 + loss2

        # PP*(right) = right calculate
        Pv_I_left = model.p_verse(input_right)
        Pv_O_left = model.p_verse(output_right)

        PPv_I_right = model.p(Pv_I_left)
        PPv_O_right = model.p(Pv_O_left)

        PPv_inputs = input_seq_dct
        PPv_inputs[:, rightdim, :] = PPv_I_right
        PPv_outputs = output_seq_dct
        PPv_outputs[:, rightdim, :] = PPv_O_right

        # calculate right loss and backward

        _, lossa = loss_funcs.mpjpe_error_p3d(PPv_inputs, all_seq[:, 0:input_n, :], dct_n, dim_used)
        _, lossb = loss_funcs.mpjpe_error_p3d(PPv_outputs, all_seq[:, input_n:(input_n + output_n), :], dct_n, dim_used)
        loss_right = lossa + lossb

        loss = loss_left + loss_right + loss_left + loss_right



        # calculate loss and backward
        loss = loss1R+loss2L

        num += 1
        plotter.plot('loss', 'train', 'Class Loss', num, loss.item())


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
    return lr_now, t_l.avg, num
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

        outputs = model.g(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used_len = len(dim_used)

        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_3d = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                   seq_len).transpose(1,
                                                                                                                      2)
        _, test_loss = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dct_n, dim_used)
        plotter.plot('loss', 'test', 'Class Loss', i, test_loss.item())
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

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_l / N, t_3d / N
#
#
def val(train_loader, model, is_cuda=False, dim_used=[], dct_n=15,rightdim=[], leftdim=[]):
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()

        outputs = model.g(inputs)

        n, _, _ = all_seq.data.shape

        _,m_err = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dct_n, dim_used)
        plotter.plot('loss', 'Val', 'Class Loss', i, m_err.item())

        # update the training loss
        t_3d.update(m_err.item() * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d.avg