# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd
from matplotlib import pyplot as plt
import logging.handlers

import argparse
import os
import random

import numpy as np
import torch
import sys
sys.path.append("./")
from mmcv import Config
from engineer.models.builder import build_backbone
import engineer.utils.logging as logging
from engineer import __version__,__author__
import engineer.utils.misc as misc
from engineer.datasets.builder import build_dataset
from engineer.core.train import train_model
import os


from engineer.utils import loss_funcs, utils as utils
from engineer.utils.opt import Options
from engineer.utils.h36motion import H36motion
import engineer.utils.data_utils as data_utils
import engineer.utils.viz as viz

logger = logging.get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Motion GCN Module')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--launcher',default='none',type=str)
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        raise NotImplementedError("distributed Training is not necessary Here")
    cfg.checkpoints = os.path.join(cfg.checkpoints,cfg.name)

    if not os.path.exists(cfg.checkpoints):
        os.mkdir(cfg.checkpoints)


    logging.setup_logging()

    logger.info('Distributed training: {}'.format(distributed))
    logger.info('GCN_Motion Version: {}\t Author: {}'.format(__version__,__author__))
    logger.info('Config: {}'.format(cfg.text))
    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    #build model and print model info
    model = build_backbone(cfg.model)
    misc.log_model_info(model)

    #optimizer build
    #follow the paper optimize we use here
    if cfg.optim_para['optimizer']['type'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim_para['optimizer']['lr'])

    #datasets build
    test_datasets=dict()
    for act in cfg.actions['all']:
        cfg.data.test.actions=act
        test_datasets[act] = build_dataset(cfg.data.test)
    val_dataset = build_dataset(cfg.data.val)
    train_dataset = build_dataset(cfg.data.train)
    logger.info(">>> data loaded !")
    logger.info(">>> train data {}".format(train_dataset.__len__()))
    logger.info(">>> validation data {}".format(val_dataset.__len__()))

    model.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    for act in cfg.acts:
        for i, (inputs, targets, all_seq) in enumerate(test_datasets[act]):
            inputs = Variable(inputs).float()
            all_seq = Variable(all_seq).float()
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                inputs = inputs.cuda(cfg.cuda_num)
                all_seq = all_seq.cuda(cfg.cuda_num)

            outputs = model(inputs)

            n, seq_len, dim_full_len = all_seq.data.shape
            dim_used_len = len(dim_used)

            _, idct_m = data_utils.get_dct_matrix(seq_len)
            idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
            outputs_t = outputs.view(-1, seq_len).transpose(0, 1)
            outputs_exp = torch.matmul(idct_m, outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                            seq_len).transpose(1, 2)
            pred_expmap = all_seq.clone()
            dim_used = np.array(dim_used)
            pred_expmap[:, :, dim_used] = outputs_exp
            targ_expmap = all_seq
            pred_expmap = pred_expmap.cpu().data.numpy()
            targ_expmap = targ_expmap.cpu().data.numpy()
            for k in range(8):
                plt.cla()
                figure_title = "action:{}, seq:{},".format(act, (k + 1))
                viz.plot_predictions(targ_expmap[k, :, :], pred_expmap[k, :, :], fig, ax, figure_title)
                plt.pause(1)


if __name__ == "__main__":
    main()
