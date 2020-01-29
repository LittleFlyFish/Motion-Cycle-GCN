import copy
from .pipelines import Compose
from .registry import DATASETS
import numpy as np
import json

from torch.utils.data import Dataset
import numpy as np
from engineer.utils import data_utils



@DATASETS.register_module
class Hm36Dataset_3d(Dataset):

    def __init__(self, path_to_data, actions, pipeline,input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = dct_used
        self.actions = actions
        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        # subs = np.array([[1], [5], [11]])
        acts = data_utils.define_actions(actions)
        subjs = subs[split]
        self.pipeline = Compose(pipeline)
        # loader data is in here
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 input_n + output_n)
        self.dim_used = dim_used

        self.all_seqs,self.input_dct_seq,self.output_dct_seq = self.pipeline(dict(all_seqs=all_seqs,dim_used=dim_used,input_n=input_n,output_n=output_n,dct_used=dct_used))


    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]
    def __repr__(self):
        return "{} @action {}".format(__class__.__name__,self.actions)


@DATASETS.register_module
class Hm36Dataset_3d_ST(Dataset):

    def __init__(self, path_to_data, actions, pipeline,input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = dct_used
        self.actions = actions
        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        # subs = np.array([[1], [5], [11]])
        acts = data_utils.define_actions(actions)
        subjs = subs[split]
        self.pipeline = Compose(pipeline)
        # loader data is in here
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 input_n + output_n)
        self.dim_used = dim_used

        self.all_seqs,self.input_dct_seq,self.output_dct_seq = self.pipeline(dict(all_seqs=all_seqs,dim_used=dim_used,input_n=input_n,output_n=output_n,dct_used=dct_used))

        ################################################################################################################
        ## change the output version to be [batch, 3, frame_n, node_n]
        all_seqs = all_seqs[:, :, dim_used]
        batch, frame_n, _ = all_seqs.shape
        node_n = int(len(dim_used)/3)

        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        self.input = np.resize(all_seqs[:, i_idx, :], (batch, input_n+output_n, 3, node_n)) ## this line of view is not sure
        self.input = np.transpose(self.input, (0, 2, 1, 3))
        self.output = np.resize(all_seqs, (batch, 3, input_n+output_n, node_n))


    def __len__(self):
        return np.shape(self.input)[0]

    def __getitem__(self, item):
        return self.input[item], self.output[item], self.all_seqs[item]
    def __repr__(self):
        return "{} @action {}".format(__class__.__name__,self.actions)

@DATASETS.register_module
class Hm36Dataset_3d_ST2(Dataset):

    def __init__(self, path_to_data, actions, pipeline,input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = dct_used
        self.actions = actions
        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        # subs = np.array([[1], [5], [11]])
        acts = data_utils.define_actions(actions)
        subjs = subs[split]
        self.pipeline = Compose(pipeline)
        # loader data is in here
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 input_n + output_n)
        self.dim_used = dim_used

        self.all_seqs,self.input_dct_seq,self.output_dct_seq = self.pipeline(dict(all_seqs=all_seqs,dim_used=dim_used,input_n=input_n,output_n=output_n,dct_used=dct_used))

        ################################################################################################################
        ## change the output version to be [batch, 3, frame_n, node_n]
        all_seqs = all_seqs[:, :, dim_used]
        batch, frame_n, _ = all_seqs.shape
        node_n = int(len(dim_used)/3)

        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n))
        self.input = np.resize(all_seqs[:, i_idx, :], (batch, input_n+output_n, 3, node_n)) ## this line of view is not sure
        self.input = np.transpose(self.input, (0, 2, 1, 3))

        self.padding_seq = np.resize(all_seqs[:, pad_idx, :], (batch, input_n+output_n, 3, node_n)) ## this line of view is not sure
        self.padding_seq = np.transpose(self.padding_seq, (0, 2, 1, 3))

        self.output = np.resize(all_seqs, (batch, 3, input_n+output_n, node_n))

        self.output = self.output[:, :, input_n: (input_n + output_n), :]


    def __len__(self):
        return np.shape(self.input)[0]

    def __getitem__(self, item):
        return self.input[item], self.padding_seq[item], self.all_seqs[item]
    def __repr__(self):
        return "{} @action {}".format(__class__.__name__,self.actions)

@DATASETS.register_module
class VideoDataset(Dataset):
    """A PyTorch video dataset for action recognition.

    This class is useful to load video data with multiple decode methods
    and applies pre-defined data pipeline with data augmentations (Normalize,
    MultiScaleCrop, Flip, etc.) and formatting operations (ToTensor, Collect,
    etc.) to return a dict with required values.

    Inputs:
        - ann_file (str): Path to an annotation file which store video info.
        - pipeline (list[dict | callable class]):
            A sequence of data augmentations and formatting operations.
        - data_prefix (str): Path to a directory where videos are held.
        - shorter_edge (int): shorter edge length of input videos.
        - input_size (int): width, height of input images
        - num_segments (int): number of extra frame segments
        - test_mode: store True when building test dataset.

    Annotation format:
        ['video_path' 'video_label'] format for each line
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 shorter_edge=256,
                 input_size=224,
                 num_segments=1,
                 test_mode=False,
                 num_classes = 101):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.shorter_edge = shorter_edge
        self.input_size = input_size
        self.num_segments = num_segments
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)



        self.video_infos,self.video_labels,self.groups = self.load_annotations()
        cnt = [0,0]
        for label in self.video_labels:
            cnt[label]+=1


    def load_annotations(self):
        #
        with open(self.ann_file, 'r') as fin:
            obj_streams = json.load(fin)

        return self.decode_streams(self.data_prefix,obj_streams)

    def prepare_train_frames(self, idx):
        imgs = copy.deepcopy(self.video_infos[idx])
        label = copy.deepcopy(self.video_labels[idx])
        group  = self.groups[idx]
        results  = dict(filenames = imgs,label= label,group = group)
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_frames(idx)
        else:
            return self.prepare_train_frames(idx)
