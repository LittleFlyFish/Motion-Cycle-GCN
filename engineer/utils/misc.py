#!/usr/bin/env python3
# Copyright (c) Sensetime, Inc. and its affiliates. All Rights Reserved.
# Author: Lingteng Qiu
import logging
import math
import numpy as np
import os
from datetime import datetime
import torch

import engineer.utils.logging as logging

logger = logging.get_logger(__name__)




def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model):
    """
    Computes the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Computes the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def log_model_info(model):
    """
    Logs info, includes number of parameters and gpu usage.
    Args:
        model (model): model to log the info.
    """
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info("nvidia-smi")
    os.system("nvidia-smi")