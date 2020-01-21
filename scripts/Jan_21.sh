#######################################################################################################################
#### today just evaluate the best situation for Cycle Case
#### evaluate different batch norm, different lr, different batch size
#### evaluate Input=25 and Output=25 version
#### evaluate the best backbone for P
#### evaluate different backbone for G

python3 tools/train_cycle.py --config ./configs/Motion_GCN_I10_O10_D15_GG\*.py

#### evaluate when loss = L1
python3 tools/train_net_l1.py --config ./configs/Motion_GCN_I10_O10_D15_G.py   # train G for inputs=10. loss =l1,

### compare of BatchNorm, LayerNorm, and GroupNorm
python3 tools/train_net.py --config ./configs/NoNorm_I10_O10_D15_G.py    # train G without BatchNorm for inputs=10

### train all the configs with Input = 25 frames and Output = 25 frames, and Dct_n = 15
python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_G.py  # train G for inputs=25
python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_G\*.py   # train G* for inputs=25


### compare the batch size = 16, 32, 64, compare the lr = 0.001,  0.0005, 0.0001, 0.00005, 0.00001
python3 tools/train_net.py --config ./configs/Motion_GCN_I10_O10_D15_G.py ## just change the configs setting

-------------------------------------------------------------------------------------------------------------------

###Plan : G use attention/ attention + gcn
python3 tools/train_net.py --config ./configs/Attention_I10_O10_D15_G.py    # train G with Attention for Input = 10

## Plan: ST-GCN auto encoder graph, + residual link + U-net link + attention + upsample and downsample



###Plan : Multi-scale skills

###Plan L_2,1 Loss