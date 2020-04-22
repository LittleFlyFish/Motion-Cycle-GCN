from .builder import build_dataset
from .hm36_dataset import Hm36Dataset_3d, H36motion
from .hm36_dataset import Hm36Dataset_3d_ST # input is padding 20 frames, output is allseqs, [batch, 3, 20, 22]
from .hm36_dataset import Hm36Dataset_3d_ST2 # input 10 frame, output 10 frame, [batch, 3, 10, 22]
from .hm36_dataset import Hm36Dataset_K     # K = 5, 5 frames turns into 3 dct_n    input [batch, 66, 60]
from .hm36_dataset import Hm36Dataset_3dLabel # Return four items includes Labels, [batch]
from .hm36_dataset import Hm36Dataset_seq2seq # For K =5 as well,  # input = [seq_len, batch, input_size], target = [seq_len, batch, target_size]
from .hm36_dataset import Hm36Dataset_3d_trans # Input = [batch, 10, 66], target = [batch, 10, 66], padding = [batch, 10, 66]
from .d3p_dataset import Pose3dPW3D, Pose3dPW # Input = [batch, 10, 66], target = [batch, 10, 66], padding = [batch, 10, 66]
from .CMU_dataset import CMU_Motion, CMU_Motion3D # Input = [batch, 10, 66], target = [batch, 10, 66], padding = [batch, 10, 66]


__all__ = ['build_dataset','Hm36Dataset_3d', 'Hm36Dataset_3d_ST', 'Hm36Dataset_3d_ST2', 'Hm36Dataset_K',
           "Hm36Dataset_3dLabel", "Hm36Dataset_seq2seq", "Hm36Dataset_3d_trans", 'Pose3dPW3D', 'Pose3dPW',
           'CMU_Motion3D', 'H36motion']