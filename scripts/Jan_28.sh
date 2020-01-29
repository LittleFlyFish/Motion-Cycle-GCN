###Plan : G use attention/ attention + gcn
python3 tools/train_net.py --config ./configs/Attention_I10_O10_D15_G.py    # train G with Attention for Input = 10
python3 tools/train_net.py --config ./configs/Attention_GCN_I10_O10_D15_G.py     # train G with Attention+GCN for Input = 10

###Plan : Dense_GCN based network
python3 tools/train_net.py --config ./configs/Dense_GCN_I10_O10_D15_G.py

## Plan: ST-GCN auto encoder graph, + residual link + U-net link + attention + upsample and downsample
python3 tools/train_ST_GCN.py --config ./configs/ST_GCN_I10_O10_D15_G.py   ## ST_GCN layer Dense structure framework
python3 tools/train_ST_GCN.py --config ./configs/ST_A_I10_O10_D15_G.py   ## ST_GCN layer conv structure framework

python3 tools/train_ST_GCN2.py --config ./configs/ST_GCN_I10_O10_D15_G2.py   ## ST_GCN layer Dense structure framework
python3 tools/train_ST_GCN2.py --config ./configs/ST_A_I10_O10_D15_G2.py   ## ST_GCN layer conv structure framework
python3 tools/train_ST_GCN2.py --config ./configs/ST_B_I10_O10_D15_G2.py
###Plan : Multi-scale skills

###Plan L_2,1 Loss