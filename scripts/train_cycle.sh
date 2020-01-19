python3 tools/train_cycle.py --config ./configs/Motion_GCN_I10_O10_D15_GG\*.py

python3 tools/train_P.py --config ./configs/P_GCN_I10_O10_D5.py  # train P

python3 tools/train_P.py --config ./configs/Inverse_P_GCN_I10_O10_D5.py  # train P*

python3 tools/train_PCycle.py --config ./configs/PCycle_GCN_I10_O10_D5.py  # train PCycle

python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_G.py   # train G for inputs=25

python3 tools/train_net.py --config ./configs/Motion_GCN_I25_O25_D15_G\*.py   # train G for inputs=25

python3 tools/train_recycle.py --config ./configs/Recycle_G101015P10105.py   # train Recycle

python3 tools/train_net.py --config ./configs/Motion_GCN_I10_O10_D15_G.py    # train the original G