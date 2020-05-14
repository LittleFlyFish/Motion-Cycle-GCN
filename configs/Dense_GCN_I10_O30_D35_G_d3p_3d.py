import numpy as np
model = dict(
    type='Dense_GCN',
    input_feature=35,
    hidden_feature=256,
    p_dropout=0.5,
    num_stage=8,
    node_n=69
)
dataset_type = 'Pose3dPW3D'
data_root = './engineer/datasets/D3P'

cuda_num = 'cuda:1'
flag = 'Dense+d3p_3D'


train_pipeline = [
    dict(type='SampleFrames', direction = True),
]

val_pipeline = [
    dict(type='SampleFrames', direction = True),
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,

    val=dict(
        type=dataset_type,
        path_to_data=data_root,
        input_n=10,
        output_n=30,
        dct_n=35,
        split=2,
    ),
    train=dict(
        type=dataset_type,
        path_to_data=data_root,
        input_n=10,
        output_n=30,
        dct_n=35,
        split=0,
    ),
    test=dict(
        type=dataset_type,
        path_to_data=data_root,
        input_n=10,
        output_n=30,
        dct_n=35,
        split=1,
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

