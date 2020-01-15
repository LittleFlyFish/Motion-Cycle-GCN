'''
why we call it container?
Because I think load image like a full Boxes, we add our data to it.
This process perhaps like container.
'''
import cv2
import numpy as np
try:
    from engineer.utils.mc_reader import MemcachedReader
except:
    pass
from ..registry import PIPELINES
import io
from engineer.utils import data_utils

@PIPELINES.register_module
class SampleFrames:
    """Sample frames from the video.

    Required keys are "filename",
    added or modified keys are "total_frames" and "frame_inds",
                               "frame_interval" and "num_clips".

    Attributes:
        direction (bool): positive or negative
        if positive:
            train G
        else:
            train G*
    """

    def __init__(self,direction):
        self.direction = direction


    def __call__(self,results):
        all_seqs=results['all_seqs']
        dim_used=results['dim_used']
        input_n=results['input_n']
        output_n=results['output_n']
        dct_used=results['dct_used']
        if self.direction :
            ori_seqs = all_seqs.copy()
            all_seqs = all_seqs[:, :, dim_used]
        else:
            #reverse the searials
            ori_seqs = all_seqs[:,::-1,...].copy()
            all_seqs = all_seqs[:,::-1,dim_used]

        all_seqs = all_seqs.transpose(0, 2, 1)
        all_seqs = all_seqs.reshape(-1, input_n + output_n)
        all_seqs = all_seqs.transpose()
        dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
        dct_m_out, _ = data_utils.get_dct_matrix(input_n + output_n)

        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)


        input_dct_seq = np.matmul(dct_m_in[0:dct_used, :], all_seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        # input_dct_seq = input_dct_seq.reshape(-1, len(dim_used) * dct_used)
        output_dct_seq = np.matmul(dct_m_out[0:dct_used, :], all_seqs)
        output_dct_seq = output_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        return ori_seqs,input_dct_seq,output_dct_seq
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str +="(direction = {}, )".format(self.direction)
        return repr_str
