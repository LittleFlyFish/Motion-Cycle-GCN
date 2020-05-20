## Download dataset from dropbox, and put under the datasets dir


#################################D3P

python3 tools/train_net.py --config ./configs/Dense_GCN_I10_O10_D15_G.py # H3.6M3D
python3 tools/train_net_cmu.py --config ./configs/Dense_GCN_I10_O10_D15_G_cmu.py
# the cuda_num setting should be the same as CMU_dataset.py line 45
python3 tools/train_net_cmu3d.py --config ./configs/Dense_GCN_I10_O10_D15_G_cmu3d.py
# the cuda_num setting should be the same as CMU_dataset.py line 98
python3 tools/train_hm.py --config ./configs/Dense_GCN_I10_O10_D15_G_hm.py    #H3.6M_angle
# the cuda_num setting should be the same as hm36_dataset.py line 46
python3 tools/train_net_d3p.py --config ./configs/Dense_GCN_I10_O30_D35_G_d3p.py
python3 tools/train_net_d3p_3d.py --config ./configs/Dense_GCN_I10_O30_D35_G_d3p_3d.py

# python3 tools/demo.py --config ./configs/Dense_GCN_I10_O10_D15_G_hm.py

# export PYTHONPATH="/home/yanran/Yanran/Projects/Motion-Cycle-GCN/engineer/utils/":"${PYTHONPATH}"

# export PYTHONPATH="/home/yanran/Envs/MP_GCN/bin/python3":"${PYTHONPATH}"

# export PYTHONPATH=$PYTHONPATH:/home/yanran/Envs/MP_GCN/lib/python3.6/site-packages