from __future__ import print_function, division
from typing import Optional, List, Union
import numpy as np
import json
import random
import os

import torch
import torch.utils.data
from scipy.ndimage import zoom

from . import VolumeDataset
from ..augmentation import Compose
from ..utils import *

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]


class MultiVolumeDataset(torch.utils.data.Dataset):
    r"""load different volume in period to balance time and memory

    Args:
        chunk_num (list): volume spliting parameters in :math:`(z, y, x)` order. Default: :math:`[2, 2, 2]`
        chunk_ind (list): predefined list of chunks. Default: `None`
        chunk_ind_split (list): rank and world_size for spliting chunk_ind in multi-processing. Default: `None`
        chunk_iter (int): number of iterations on each chunk. Default: -1
        chunk_stride (bool): allow overlap between chunks. Default: `True`
        volume_json (str): json file for input image. Default: ``'path/to/image'``
        label_json (str, optional): json file for label. Default: `None`
        valid_mask_json (str, optional): json file for valid mask. Default: `None`
        mode (str): ``'train'``, ``'val'`` or ``'test'``. Default: ``'train'``
        pad_size (list): padding parameters in :math:`(z, y, x)` order. Default: :math:`[0,0,0]`
    """

    def __init__(self,
                 volume_path,
                 label_path,
                 mode: str = 'train',
                 load_2d=False,
                 chunk_iter: int = -1,
                 pad_size: List[int] = [0, 0, 0],
                 pad_mode: str = 'replicate',
                 **kwargs):

        self.kwargs = kwargs
        self.mode = mode
        self.volume_path = volume_path
        self.label_path = label_path
        self.volume_done = []
        self.volume_sample = None
        self.read_fn = readvol if not load_2d else readimg_as_vol
        self.pad_size = pad_size
        self.pad_mode = pad_mode
        self.label_sample = None
        self.dataset = None
        self.chunk_iter = chunk_iter

    def updatechunk(self, do_load=True):
        r"""Update the coordinates to a new chunk in the large volume.
        """
        if len(self.volume_done) == len(self.volume_path):
            self.volume_done = []
        volume_rest = list(set(self.volume_path) - set(self.volume_done))
        if self.mode == 'train':
            self.volume_sample = volume_rest[int(np.floor(random.random() * len(volume_rest)))]
            index = self.volume_path.index(self.volume_sample)
            if self.label_path is not None:
                self.label_sample = self.label_path[index]
        self.volume_done += [self.volume_sample]

        if do_load:
            self.loadchunk()

    def loadchunk(self):
        r"""Load the chunk and construct a VolumeDataset for processing.
        """
        try:
            rank = int(os.environ["LOCAL_RANK"])
        except:
            print('DEBUG without distributed training. Load volume: ', self.volume_sample)
        else:
            print(rank, 'load chunk: ', self.volume_sample)
        volume = self.read_fn(self.volume_sample)
        # intensity normalization: is done before saving chunks
        # volume = normalize_z(volume, clip=True)
        # spacial normalization
        # volume = maybe_scale(volume)
        # label = self.read_fn(self.label_sample)
        # assert (volume.shape == label.shape)
        volume = [np.pad(volume, get_padsize(self.pad_size), self.pad_mode)]
        # label = [np.pad(label, get_padsize(self.pad_size), self.pad_mode)]
        label = volume
        self.dataset = VolumeDataset(volume, label, valid_mask=None, mode=self.mode,
                                     # specify chunk iteration number for training (-1 for inference)
                                     iter_num=self.chunk_iter if self.mode == 'train' else -1,
                                     **self.kwargs)

    def maybe_scale(self, data, scale, order=0):
        if (np.array(scale) != 1).any():
            for i in range(len(data)):
                dt = data[i].dtype
                data[i] = zoom(data[i], scale,
                               order=order).astype(dt)

        return data
