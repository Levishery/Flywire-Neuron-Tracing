from __future__ import print_function, division
from typing import Optional, List, Union
import numpy as np
import json
import random
import os

import torch
import torch.utils.data
from scipy.ndimage import zoom
from cloudvolume import CloudVolume

from . import VolumeDataset
from . import ConnectorDataset
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
                 connector_dataset=False,
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
        self.connector_dataset = connector_dataset
        if self.connector_dataset:
            self.vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0')
            self.block_image_path = '/braindat/lab/liusl/flywire/block_data/fafbv14'

    def updatechunk(self, do_load=True):
        r"""Update the coordinates to a new chunk in the large volume.
        """
        if self.mode == 'train':
            if len(self.volume_done) == len(self.volume_path):
                self.volume_done = []
        volume_rest = list(set(self.volume_path) - set(self.volume_done))
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
        if self.connector_dataset:
            # self.volume_sample = '/braindat/lab/liusl/flywire/block_data/train_30/connector_15_10_123.csv'
            block_index = self.volume_sample.split('/')[-1].split('.')[0]
            block_xyz = block_index.split('_')[1:]
            img_volume_path = os.path.join(self.block_image_path, block_index, '*.png')
            volume = self.read_fn(img_volume_path)
            volume = [volume[12:70, :, :]]
            # volume = volume[29:55, 156:-156, 156:-156]
            cord_start = self.xpblock_to_fafb(int(block_xyz[2]), int(block_xyz[1]), int(block_xyz[0]), 28, 0, 0)
            cord_end = self.xpblock_to_fafb(int(block_xyz[2]), int(block_xyz[1]), int(block_xyz[0]), 53, 1735, 1735)
            volume_ffn1 = self.vol_ffn1[cord_start[0]/4-156:cord_end[0]/4+156+1,
                             cord_start[1]/4-156:cord_end[1]/4+156+1, cord_start[2]-16:cord_end[2]+16+1]
            # volume_ffn1 = self.vol_ffn1[cord_start[0] / 4:cord_end[0] / 4 + 1,
            #               cord_start[1] / 4:cord_end[1] / 4 + 1, cord_start[2]:cord_end[2] + 1]
            volume_ffn1 = np.transpose(volume_ffn1.squeeze(), (2, 0, 1))
            self.dataset = ConnectorDataset(connector_path=self.volume_sample, volume=volume,
                                            vol_ffn1=volume_ffn1, mode=self.mode,
                                            iter_num=self.chunk_iter if self.mode =='train' else -1, **self.kwargs)
        else:
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

    def xpblock_to_fafb(self, z_block, y_block, x_block, z_coo=0, y_coo=0, x_coo=0):
        '''
        函数功能:xp集群上的点到fafb原始坐标的转换
        输入:xp集群点属于的块儿和块内坐标,每个块的大小为1736*1736*26(x,y,z顺序)
        输出:fafb点坐标
        z_block:xp集群点做所在块儿的z轴序号
        y_block:xp集群点做所在块儿的y轴序号
        x_clock:xp集群点做所在块儿的x轴序号
        z_coo:xp集群点所在块内的z轴坐标
        y_coo:xp集群点所在块内的y轴坐标
        x_coo:xp集群点所在块内的x轴坐标

        '''

        # 第二个：原来的,z准确的,x,y误差超过5,block大小为 26*1736*1736
        # x_fafb = 1736 * 4 * x_block - 17830 + 4 * x_coo
        x_fafb = 1736 * 4 * x_block - 17631 + 4 * x_coo
        # y_fafb = 1736 * 4 * y_block - 19419 + 4 * y_coo
        y_fafb = 1736 * 4 * y_block - 19211 + 4 * y_coo
        z_fafb = 26 * z_block + 15 + z_coo
        # z_fafb = 26 * z_block + 30 + z_coo
        return (x_fafb, y_fafb, z_fafb)

