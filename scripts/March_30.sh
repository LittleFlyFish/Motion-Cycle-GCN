python3 tools/train_smooth.py --config ./configs/Motion_GCN_I10_O10_D15_G_smooth.py
python3 tools/train_conv.py --config ./configs/Motion_GCN_I10_O10_D15_G_conv.py
python3 tools/train_net.py --config ./configs/Motion_GCN_I10_O10_D15_G.py
python3 tools/train_net.py --config ./configs/ST_GCNA_I10_O10_D15_G.py

python3 tools/train_conv.py --config ./configs/Dense_GCN_I10_O10_D15_G.py


python3 tools/train_offsetloss.py --config ./configs/Motion_GCN_I10_O10_D15_G_offset.py