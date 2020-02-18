from .Motion_GCN import Motion_GCN
from .SemGCN import SemGCN
from .Cycle_GCN import Cycle_GCN
from .Cycle_P import Cycle_P
from .P_GCN import P_GCN
from .Recycle_GCN import Recycle_GCN
from .G_NoNorm import G_NoNorm
from .G_Attention import G_Attention
from .G_Attention_GCN import G_Attention_GCN
from .Dense_GCN import Dense_GCN
from .Fuse_GCN import Fuse_GCN # ST-GCN as encoder, Motion_GCN as decoder
from .ST_GCN_Dense import ST_GCN_Dense
from .ST_A import ST_A
from .ST_B import ST_B
from .ST_C import ST_C
from .NewGCN import NewGCN
from .GCNGRU import GCNGRU
from .ST_D import ST_D
from .ST_E import ST_E
from .Multi_GCN import Multi_GCN
from .K_GCN import K_GCN
from .GCN_2task import GCN_2task
from .Seq2Seq import Seq2Seq
__all__=["Motion_GCN","SemGCN","Cycle_GCN", "P_GCN", "Cycle_P", "Recycle_GCN", "G_NoNorm",
         "G_Attention", "G_Attention_GCN", "Dense_GCN", "Fuse_GCN", "ST_GCN_Dense",
         "ST_A", "ST_B", "ST_C", "NewGCN", "GCNGRU", "ST_D", "ST_E", "Multi_GCN", "K_GCN",
         "GCN_2task", "Seq2Seq"]

# ST_A : 5 layer ST_GCN as encoder, ST_GCN as decoder, only downsample on frames
# ST_B: ST_GCN autoencoder, only 2 downsample upsample layers
# ST_C:  ST_GCN + Motion_GCN + ST_GCN
# ST_D: GCN autoencoder, only 2 downsample upsample layers.
# ST_E: ST_GCN long term + short term encoder, predict next frame recursively
# ST_GCN_Dense: Dense network with each layer ST_GCN
# New GCN: a dense connected but RNN similar structure network, each layer GCN
# GCNGRU: GRU structure with each operator GCN.
