import numpy as np
edge =([0,1],[1,2],[2,3],[0,8],[8,4],[4,5],[5,6],[6,7],\
      [8,9],[9,10],[10,11],\
      [8,12],[12,13],[13,14],[14,15],[14,16],\
      [8,17],[17,18],[18,19],[19,20],[19,21]
      )

train_pipeline = [
    dict(type='SampleFrames',direction = True),
]

val_pipeline = [
    dict(type='SampleFrames', direction=True),
]
model = dict(
    type='SemGCN',
    hid_dim = 256,
    adj=[22,edge],
    coords_dim=(15,15),
    num_layers=4,
    nodes_group=None,
    p_dropout=0.5
)
dataset_type = 'Hm36Dataset_3d'
data_root = './datasets/h3.6m/dataset'
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=0,
    val=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions="all",
        input_n=10,
        output_n=10,
        dct_used=15,
        split=2,
        sample_rate=2,
        pipeline=val_pipeline
    ),
    train=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions="all",
        input_n=10,
        output_n=10,
        dct_used=15,
        split=0,
        sample_rate=2,
        pipeline=train_pipeline
    ),
    test=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions=None,
        input_n=10,
        output_n=10,
        dct_used=15,
        split=1,
        sample_rate=2,
        pipeline = val_pipeline
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

