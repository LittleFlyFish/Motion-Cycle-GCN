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
python3 tools/train_ST_GCN2.py --config ./configs/ST_B_I10_O10_D15_G2.py  ## DownSample Upsample Autoencoder
python3 tools/train_ST_GCN2.py --config ./configs/ST_C_I10_O10_D15_G2.py  ## ST instead of dct feature
python3 tools/train_ST_GCN.py --config ./configs/ST_C_I10_O10_D15_G.py
python3 tools/train_NewGCN.py --config ./configs/NewGCN_I10_O10_h32f16.py   ## test new GCN feature
python3 tools/train_NewGCN.py --config ./configs/NewGCN_I10_O10_h32f32.py
python3 tools/train_NewGCN.py --config ./configs/NewGCN_I10_O10_h64f128.py
python3 tools/train_NewGCN.py --config ./configs/NewGCN_I10_O10_h64f64.py

python3 tools/train_ST_GCN.py --config ./configs/ST_GCN_I10_O10_D15_G.py

python3 tools/train_NewGCN.py --config ./configs/GCNGRU_I10_O10_h128f128.py

python3 tools/train_ST_GCN2.py --config ./configs/ST_B_I10_O10_D15_G2_h256.py

python3 tools/train_net.py --config ./configs/Motion_GCN_I10_O10_D15_G_n3.py

python3 tools/train_net.py --config ./configs/ST_D_I10_O10_D15_G.py

python3 tools/train_ST_GCN2.py --config ./configs/ST_E_I10_O10_D15_G2.py

python3 tools/train_ST_GCN2.py --config ./configs/ST_E_I50_O10_D15_G2.py

python3 tools/train_ST_GCN.py --config ./configs/ST_B_I10_O10_D15_G.py

python3 tools/train_net.py --config ./configs/Multi_GCN_I10_O10_D15_G.py

##############################################################################
# feature size test
python3 tools/train_NewGCN.py --config ./configs/NewGCN_I10_O10_h8f8.py

-------------------------------------------------------------------------------------------
python3 tools/train_K.py --config ./configs/K_GCN_I10_O10_D15_G.py

python3 tools/train_K.py --config ./configs/K_GCN_I10_O10_D15_Goutput.py

python3 tools/train_2task.py --config ./configs/GCN_2task_I10_O10_D15_G.py

python3 tools/train_2task.py --config ./configs/GCN_2task_I10_O10_D15_GAtt.py

python3 tools/train_seq2seq.py --config ./configs/Seq2Seq_I10_O10_D15_K5.py

python3 tools/train_trans.py --config ./configs/Trans_I10_O10_D15_G.py

python3 tools/train_multi.py --config ./configs/Multi_GCN_I10_O10_D15_G.py

python3 tools/train_multi.py --config ./configs/Multi_GCN_I10_O10_D15_PLUS.py



###Plan : Multi-scale skills

###Plan L_2,1 Loss