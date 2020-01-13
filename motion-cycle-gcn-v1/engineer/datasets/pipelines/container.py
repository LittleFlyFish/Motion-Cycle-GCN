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

@PIPELINES.register_module
class SampleFrames:
    """Sample frames from the video.

    Required keys are "filename",
    added or modified keys are "total_frames" and "frame_inds",
                               "frame_interval" and "num_clips".

    Attributes:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
        num_clips (int): Number of clips to be sampled.
        temporal_jitter (bool): Whether to apply temporal jittering.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.reader = MemcachedReader()

    def _sample_clips(self, num_frames):
        """
        Choose frame indices for the video.

        Calculate the average interval for selected frames, and randomly
        shift them within offsets between [0, avg_interval]. If the total
        number of frames is smaller than clips num or origin frames length,
        it will return all zero indices.

        Args:
            num_frames: total number of frame in the video.

        Returns: list of sampled frame indices
        """
        ori_clip_len = self.clip_len * self.frame_interval

        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        else:
            clip_offsets = np.zeros((self.num_clips, ))

        return clip_offsets

    def __call__(self, results):
        img_inds = results['filenames']
        total_frames = len(img_inds)
        results['total_frames'] = total_frames

        clip_offsets = self._sample_clips(total_frames)

        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                0, self.frame_interval, size=self.clip_len)
            frame_inds += perframe_offsets

        frame_inds = np.mod(frame_inds, total_frames)
        results['frame_inds'] = frame_inds
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips

        return results
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str +="(clip={}, frame_interval={}, num_clips={}, temporal_jitter={})".format(self.clip_len,self.frame_interval,self.num_clips,self.temporal_jitter)
        return repr_str

@PIPELINES.register_module
class OpenCVDecode:
    """Using OpenCV to decode the video

    Required keys are "filename" and "frame_inds",
    add or modified keys are "imgs" and "ori_shape".

    Attributes:
        multi_thread (bool): If set to True, it will
            apply multi thread processing.
    """

    def __init__(self, multi_thread=False,pad_w=224,pad_h=224):
        self.multi_thread = multi_thread
        self.pad_w = pad_w
        self.pad_h = pad_h
    def _open(self,img):
        try:
            filebytes = self.reader(img)
            buff = io.BytesIO(filebytes)
            image = Image.open(buff).convert('RGB')
        except:
            image = cv2.imread(img)
            # image = Image.open(img).convert('RGB')
        return image
    def __pad_img_size(self,img):
        h,w = img.shape[0:2]
        ratio_h,ratio_w = h/self.pad_h,w/self.pad_w

        pad_img = np.zeros((self.pad_h,self.pad_w,3),np.uint8)
        pad_img[:,:,...] = [104,116,124]


        if h/ratio_w > self.pad_h:
            # according to h
            img =cv2.resize(img,(int(w/ratio_h),self.pad_h))
            new_h,new_w = img.shape[:2]
        else:
            #according to w
            img = cv2.resize(img,(self.pad_w,int(h/ratio_w)))
            new_h,new_w = img.shape[:2]
        pad_img[:new_h,:new_w,:] = img
        return pad_img

    def __call__(self, results):
        filename = results['filenames']
        frame_inds = results['frame_inds']


        imgs = list()

        if frame_inds.ndim != 1:
            frame_inds = np.squeeze(frame_inds)

        for frame_ind in frame_inds:
            # cur_content = cv2.imread(filename[frame_ind])
            cur_content = self._open(filename[frame_ind])
            pad_content = self.__pad_img_size(cur_content)
            imgs.append(pad_content)
        imgs = np.array(imgs)
        imgs = imgs.transpose([0, 3, 1, 2])
        results['imgs'] = np.array(imgs)
        results['ori_shape'] = imgs.shape[-2:]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(multi_thread={})'.format(self.multi_thread)
        return repr_str