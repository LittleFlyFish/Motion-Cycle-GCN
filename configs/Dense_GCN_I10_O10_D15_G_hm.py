import numpy as np
model = dict(
    type='Dense_GCN',
    input_feature=15,
    hidden_feature=256,
    p_dropout=0.5,
    num_stage=12,
    node_n=66
)
dataset_type = 'H36motion'
data_root = './engineer/datasets/h3.6m/dataset'

cuda_num ='cuda:1'
flag = 'Dense+HM3.6_angle'


data = dict(
    videos_per_gpu=2,
    workers_per_gpu=0,
    val=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions="all",
        input_n=10,
        output_n=10,
        dct_n=15,
        split=2,
        sample_rate=2
    ),
    train=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions="all",
        input_n=10,
        output_n=10,
        dct_n=15,
        split=0,
        sample_rate=2
    ),
    test=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions=None,
        input_n=10,
        output_n=10,
        dct_n=15,
        split=1,
        sample_rate=2
    )
)
#
# optimizer
optim_para=dict(
    optimizer = dict(type='Adam',lr=0.0005),
    lr_decay=2,
    lr_gamma= 0.96
)
total_epochs = 80
max_norm= True
checkpoints="./checkpoints"
actions=dict(all = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"],
            all_srnn= ["walking", "eating", "smoking", "discussion"]
             )
dataloader=dict(
    num_worker=10,
    batch_size=dict(
        train=16,
        test=128
    )
)
resume=dict(
    start = 0
)
name=__name__

