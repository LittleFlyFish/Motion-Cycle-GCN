import numpy as np
model = dict(
    type='Dense_GCN',
    input_feature=15,
    hidden_feature=256,
    p_dropout=0.5,
    num_stage=12,
    node_n=64
)
dataset_type = 'CMU_Motion'
data_root = './engineer/datasets/cmu_mocap'

cuda_num = 'cuda:0'
flag = 'Dense+CMU'

data = dict(
    videos_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions="all",
        input_n=10,
        output_n=10,
        dct_n = 15,
        split=0
    ),
    test=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions="all",
        input_n=10,
        output_n=10,
        dct_n = 15,
        split=1
    )
)
#
# optimizer
optim_para=dict(
    optimizer = dict(type='Adam',lr=0.0005),
    lr_decay=2,
    lr_gamma= 0.96
)
total_epochs = 50
max_norm= True
checkpoints="./checkpoints"
actions=dict(all = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
               "washwindow"]
             )
dataloader=dict(
    num_worker=4,
    batch_size=dict(
        train=16,
        test=128
    )
)
resume=dict(
    start = 0
)
name=__name__

