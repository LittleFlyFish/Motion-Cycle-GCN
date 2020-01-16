import platform
from functools import partial
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader, DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def worker_init_fn(worker_id):
    set_seed(1234 + worker_id)


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     pin_memory=True,
                     **kwargs):
    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        # collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
        **kwargs)

    return data_loader
def _dist_train(dataset,cfg,validate = False):
    raise NotImplementedError("Careful: I don't realize it now!")
def _non_dist_train(dataset,cfg,validate):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if validate :
        data_loaders = [
            build_dataloader(
                ds,
                cfg.TEST.BATCH_SIZE,
                cfg.DATA_LOADER.NUM_WORKERS,
                cfg.NUM_GPUS,
                dist=False) for ds in dataset
        ]
    else:
        data_loaders = [
            build_dataloader(
                ds,
                cfg.TRAIN.BATCH_SIZE,
                cfg.DATA_LOADER.NUM_WORKERS,
                cfg.NUM_GPUS,
                dist=False) for ds in dataset
        ]
    return data_loaders

def construct_dataloader(cfg,dataset,validate):
    if cfg.TRAIN.DISTRIBUTE:
        return _dist_train(dataset, cfg, validate=validate)
    else:
        return _non_dist_train(dataset, cfg, validate=validate)
