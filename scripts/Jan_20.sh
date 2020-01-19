# This is the plan of experiement:

### Plan A, train all the configs with Input = 25 frames and Output = 25 frames, and Dct_n = 15
python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_G.py   # train G for inputs=25

python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_G\*.py   # train G for inputs=25

python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_GG\*.py   # train GG* for inputs=25

python3 tools/train_P.py --config ./configs/P_GCN_I25_O25_D15.py  # train P for inputs=25

python3 tools/train_P.py --config ./configs/Inverse_P_GCN_I25_O25_D15.py  # train P* for inputs=25

python3 tools/train_PCycle.py --config ./configs/PCycle_GCN_I25_O25_D15.py  # train PCycle for inputs=25

python3 tools/train_recycle.py --config ./configs/Recycle_G252515P252515.py  # train PCycle for inputs=25

### Plan B, Comparison of MSE loss, L_1 loss, Recycle Loss without P constrains

####engineer.utils.loss_funcs   mpjpe_error function, change torch.norm (x, 2, 1) into torch.norm(x, 1, 1)

### train_recycle.py delete the loss_right and loss_left term and train Recycle. Only Command the G, G* load_state_dict
### use the Pretrained P part.

### Plan C, evaluate Recycle part with /without initialization
##without initialization ### command the G, G*, P, P* load_state_dict part in models.backbones——Recycle_GCN


### Plan D, compare the batch size = 16, 32, 64
# change the configs file setting of batch size, use train G and Recycle as example

### Plan E, compare the lr = 0.001,  0.0005, 0.0001, 0.00005, 0.00001

### Plan F, compare of BatchNorm, LayerNorm, and GroupNorm
## change the models.backbones.Motion_GCN nn.batchnorm1d to nn.LayerNorm