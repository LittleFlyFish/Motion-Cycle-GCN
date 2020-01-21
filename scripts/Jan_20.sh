# This is the plan of experiement:

### Note: command the load_state_dict in Cycle_GCN always when train GG*
### python -m visdom.server # after run experienment, run Visdom TO visualize the loss, and save these images

### Plan A, evaluate Recycle part with /without pretrained data
## with pretrained data
python3 tools/train_recycle.py --config ./configs/Recycle_G252515P252515.py  # train Recycle for inputs=25
python3 tools/train_recycle.py --config ./configs/Recycle_G101015P10105.py  # train Recycle for inputs=10

##without pretrained data ### command the G, G*, P, P* load_state_dict part in models.backbones——Recycle_GCN before train
python3 tools/train_recycle.py --config ./configs/Recycle_G252515P252515.py  # train Recycle for inputs=25
python3 tools/train_recycle.py --config ./configs/Recycle_G101015P10105.py  # train Recycle for inputs=10

### Plan B, Comparison of MSE loss, L_1 loss, Recycle Loss without P constrains

####enter engineer.utils.loss_funcs   mpjpe_error function, change torch.norm (x, 2, 1) into torch.norm(x, 1, 1) before train
python3 tools/train_net_l1.py --config ./configs/Motion_GCN_I10_O10_D15_G.py   # train G for inputs=10. loss =l1,
python3 tools/train_recycle.py --config ./configs/Recycle_G101015P10105.py  # train Recycle for inputs=10

### train_recycle.py delete the loss_right and loss_left term and train Recycle. Only Command the G, G* load_state_dict
### use the Pretrained P part.
python3 tools/train_recycle.py --config ./configs/Recycle_G101015P10105.py  # train Recycle for inputs=10

### Plan C, compare of BatchNorm, LayerNorm, and GroupNorm
## change the models.backbones.Motion_GCN nn.batchnorm1d to nn.LayerNorm
python3 tools/train_cycle.py --config ./configs/Motion_GCN_I10_O10_D15_GG\*.py   # train GG* for inputs=10
python3 tools/train_recycle.py --config ./configs/Recycle_G101015P10105.py  # train Recycle for inputs=10

### Plan D, train all the configs with Input = 25 frames and Output = 25 frames, and Dct_n = 15
python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_G.py   # train G for inputs=25
python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_G\*.py   # train G for inputs=25
python3 tools/train_cycle.py --config ./configs/Motion_GCN_I25_O25_D15_GG\*.py   # train GG* for inputs=25
python3 tools/train_P.py --config ./configs/P_GCN_I25_O25_D15.py  # train P for inputs=25
python3 tools/train_P.py --config ./configs/Inverse_P_GCN_I25_O25_D15.py  # train P* for inputs=25
python3 tools/train_PCycle.py --config ./configs/PCycle_GCN_I25_O25_D15.py  # train PCycle for inputs=25
python3 tools/train_recycle.py --config ./configs/Recycle_G252515P252515.py  # train Recycle for inputs=25


### Plan E, compare the batch size = 16, 32, 64
# change the configs file setting of batch size, use train G and Recycle as example
python3 tools/train_cycle.py --config ./configs/Motion_GCN_I10_O10_D15_GG\*.py   # train GG* for inputs=10
python3 tools/train_recycle.py --config ./configs/Recycle_G101015P10105.py  # train Recycle for inputs=10
python3 tools/train_cycle.py --config ./configs/Motion_GCN_I25_O25_D15_GG\*.py   # train GG* for inputs=10
python3 tools/train_recycle.py --config ./configs/Recycle_G252515P252515.py  # train Recycle for inputs=25

### Plan F, compare the lr = 0.001,  0.0005, 0.0001, 0.00005, 0.00001

