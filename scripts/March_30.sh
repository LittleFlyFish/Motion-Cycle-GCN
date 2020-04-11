python3 tools/train_smooth.py --config ./configs/Motion_GCN_I10_O10_D15_G_smooth.py
python3 tools/train_conv.py --config ./configs/Motion_GCN_I10_O10_D15_G_conv.py
python3 tools/train_net.py --config ./configs/Motion_GCN_I10_O10_D15_G.py
python3 tools/train_net.py --config ./configs/ST_GCNA_I10_O10_D15_G.py

python3 tools/train_conv.py --config ./configs/Dense_GCN_I10_O10_D15_G.py

python3 tools/train_offsetloss.py --config ./configs/Motion_GCN_I10_O10_D15_G_offset.py


# April 10th experiments
python3 tools/train_NewCycle.py --config ./configs/Motion_GCN_I10_O10_D15_G.py # window5, lamda=0.1
python3 tools/train_NewPCycle.py --config ./configs/Motion_GCN_I10_O10_D15_G.py # window4, cuda1
python3 tools/train_NewCycle.py --config ./configs/Dense_GCN_I10_O10_D15_G.py
python3 tools/train_NewPCycle.py --config ./configs/Dense_GCN_I10_O10_D15_G.py # window3, cuda2

python3 tools/train_net.py --config ./configs/Motion_GCN_I10_O10_D15_G.py # window2, cuda0

python3 tools/train_net.py --config ./configs/Subnet_GCN_I10_O10_D15_G.py #window5, cuda0
python3 tools/train_Subnet.py --config ./configs/Subnet_GCN_I10_O10_D15_G.py #windo4, cuda2