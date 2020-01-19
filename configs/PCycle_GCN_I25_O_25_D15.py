import numpy as np
model = dict(
    type= "Cycle_P",
    P=dict(    
    type='P_GCN',
    input_feature=15,
    hidden_feature=256,
    p_dropout=0.5,
    num_stage=12,
    node_n=39
    ),
    P_verse=dict(
    type='P_GCN',
    input_feature=5,
    hidden_feature=256,
    p_dropout=0.5,
    num_stage=12,
    node_n=39
    ),
    P_meta="./checkpoints/P_GCN_I10_O10_D5/ckpt_train_P_3D_in10_out10_dct_n_5_best.pth.tar",
    P_verse_meta =  "./checkpoints/Inverse_P_GCN_I10_O10_D5/ckpt_train_P_3D_in10_out10_dct_n_5_best.pth.tar"
)
dataset_type = 'Hm36Dataset_3d'
data_root = './engineer/datasets/h3.6m/dataset'
left = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16] # 22 joints in the order, left is the input of P
right = [4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21] # right is the output of P

train_pipeline = [
    dict(type='SampleFrames',direction = True),
]

val_pipeline = [
    dict(type='SampleFrames', direction=True),
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=0,
    val=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions="all",
        input_n=25,
        output_n=25,
        dct_used=15,
        split=2,
        sample_rate=2,
        pipeline=val_pipeline
    ),
    train=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions="all",
        input_n=25,
        output_n=25,
        dct_used=15,
        split=0,
        sample_rate=2,
        pipeline=train_pipeline
    ),
    test=dict(
        type=dataset_type,
        path_to_data=data_root,
        actions=None,
        input_n=25,
        output_n=25,
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