from __future__ import print_function, division
from typing import Optional, List
import numpy as np
import pandas as pd
import time
import random
from cloudvolume import CloudVolume
from PIL import Image
import matplotlib.pyplot as plt

import torch.utils.data
from ..augmentation import Compose
from ..utils import *

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]
DEBUG = 0
seg2target_factor = 1


class ConnectorDataset(torch.utils.data.Dataset):
    """
    Dataset class for connector sampling. Providing the path to the csv files of blocked connectors, the dataset first
    sample a block and load the image volume and segmentation volume into the memory. The during the iterations it
    samples the connectors in the csv file and get the corresponding image sub-volume and segmentation result.

    Args:
        volume (list): list of image volumes.
        label (list, optional): list of label volumes. Default: None
        valid_mask (list, optional): list of valid masks. Default: None
        valid_ratio (float): volume ratio threshold for valid samples. Default: 0.5
        sample_volume_size (tuple, int): model input size.
        sample_label_size (tuple, int): same as input size for training and validation.
        cropped_label_size (tuple, int): model output size, for model without padding.
        sample_stride (tuple, int): stride size for sampling.
        augmentor (connectomics.data.augmentation.composition.Compose, optional): data augmentor for training. Default: None
        target_opt (list): list the model targets generated from segmentation labels.
        weight_opt (list): list of options for generating pixel-wise weight masks.
        mode (str): ``'train'``, ``'val'`` or ``'test'``. Default: ``'train'``
        do_2d (bool): load 2d samples from 3d volumes. Default: False
        iter_num (int): total number of training iterations (-1 for inference). Default: -1
        reject_size_thres (int, optional): threshold to decide if a sampled volumes contains foreground objects. Default: 0
        reject_diversity (int, optional): threshold to decide if a sampled volumes contains multiple objects. Default: 0
        reject_p (float, optional): probability of rejecting non-foreground volumes. Default: 0.95

    """

    background: int = 0  # background label index

    def __init__(self, connector_path, volume, model_input_size, sample_volume_size, iter_num: int = -1, vol_ffn1=None,
                 mode='train',
                 data_mean=0.5, data_std=0.5,
                 augmentor: AUGMENTOR_TYPE = None,
                 target_opt: TARGET_OPT_TYPE = ['1'],
                 weight_opt: WEIGHT_OPT_TYPE = [['1']],
                 erosion_rates: Optional[List[int]] = None,
                 dilation_rates: Optional[List[int]] = None,
                 **kwargs):

        assert mode in ['train', 'val', 'test']
        self.vol_ffn1 = vol_ffn1
        self.mode = mode
        # data format
        self.sample_volume_size = np.asarray(sample_volume_size)
        self.model_input_size = model_input_size
        self.connector_path = connector_path
        self.volume = volume
        self.connector_list = pd.read_csv(self.connector_path, header=None)
        self.connector_num = len(self.connector_list)
        self.sample_num_a = len(self.connector_list)
        self.label = None
        self.valid_mask = None
        self.augmentor = augmentor
        # target and weight options
        self.target_opt = target_opt
        self.weight_opt = weight_opt
        # For 'all', users will create their own targets
        if self.target_opt[-1] == 'all':
            self.target_opt = self.target_opt[:-1]
            self.weight_opt = self.weight_opt[:-1]
        self.erosion_rates = erosion_rates
        self.dilation_rates = dilation_rates
        # normalization
        self.data_mean = data_mean
        self.data_std = data_std

        # dataset: channels, depths, rows, cols
        # volume size, could be multi-volume input
        self.model_input_size = model_input_size

        # random sample factors
        self.alpha_neg = 10
        self.alpha_offset = 10
        self.neg_point_num = 100

        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training.
        self.iter_num = max(
            iter_num, self.sample_num_a) if self.mode == 'train' else self.sample_num_a
        print('Total number of samples to be generated: ', self.iter_num)

    def __len__(self):
        # total number of possible samples
        return self.iter_num

    def __getitem__(self, index):
        # orig input: keep uint/int format to save cpu memory
        # output sample: need np.float32

        vol_size = self.model_input_size
        if self.mode == 'train':
            # self.start_time = time.perf_counter()
            if DEBUG:
                self.start_time = time.perf_counter()
            # random sample during training
            idx = random.randint(0, self.connector_num-1)
            # print(idx)
            connector = self.connector_list.iloc[idx]
            # connector = self.connector_list.iloc[386]
            return self._connector_to_target_sample(connector)

        elif self.mode == 'val':
            connector = self.connector_list.iloc[index]
            return self._connector_to_target_sample(connector)

        elif self.mode == 'test':
            connector = self.connector_list.iloc[index]
            return self._connector_to_volume_sample(connector)

    def _crop_with_pos(self, pos, vol_size):
        out_volume = (crop_volume(
            self.volume[pos[0]], vol_size, pos[1:]) / 255.0).astype(np.float32)
        return pos, out_volume

    def _connector_to_target_sample(self, connector):
        np.random.seed(random.randint(0,10))
        cord = connector[2][1:-1].split()
        cord = self.fafb_to_block(float(cord[0]), float(cord[1]), float(cord[2]))

        # sample the nearby segments as negative samples, while masking the others as background
        out_valid = None
        seg_start = connector[0]
        seg_positive = connector[1]
        neg_cord_offset = self.sample_from_normal3d(self.model_input_size[0] / self.alpha_neg, self.model_input_size[1] / self.alpha_neg, self.neg_point_num)

        # use random offset to prevent the model from using the position information
        sample_offset = self.sample_from_normal3d(self.model_input_size[0] / self.alpha_offset, self.model_input_size[1] / self.alpha_offset, 1)
        while not self.is_valid_offset(sample_offset, cord):
            sample_offset = self.sample_from_normal3d(self.model_input_size[0] / self.alpha_offset, self.model_input_size[1] / self.alpha_offset, 1)

        # the negative position should be around the connection point
        neg_cord_offset = neg_cord_offset - sample_offset
        neg_cord_offset = np.asarray([np.clip(neg_cord_offset[0], - self.model_input_size[0] / 2, self.model_input_size[0] / 2 - 1), np.clip(neg_cord_offset[1], - self.model_input_size[1] / 2, self.model_input_size[1] / 2 - 1), np.clip(neg_cord_offset[2], - self.model_input_size[2] / 2, self.model_input_size[2] / 2 - 1)])
        neg_cord_pos = neg_cord_offset + np.transpose(np.asarray([self.sample_volume_size / 2]))
        pos, out_volume = self._crop_with_pos([0, cord[2] - self.sample_volume_size[0] / 2 + sample_offset[0][0],
                                               cord[0] - self.sample_volume_size[1] / 2 + sample_offset[1][0],
                                               cord[1] - self.sample_volume_size[2] / 2 + sample_offset[2][0]], self.sample_volume_size)
        out_label = crop_volume(self.vol_ffn1, self.sample_volume_size, pos[1:])
        neg_cord_pos = list(neg_cord_pos.astype(np.int32))
        seg_negative = np.asarray(np.unique(out_label[neg_cord_pos[0], neg_cord_pos[1], neg_cord_pos[2]]))
        seg_negative = np.setdiff1d(seg_negative, [0, seg_positive, seg_start])

        data = {'image': out_volume,
                'label': out_label,
                'valid_mask': out_valid}
        augmented = self.augmentor(data)
        out_volume, out_label = augmented['image'], augmented['label']
        out_valid = augmented['valid_mask']
        pos_data = {'pos': pos,
                    'seg_start': seg_start,
                    'seg_positive': seg_positive,
                    'seg_negative': seg_negative}
        out_target = seg_to_targets(
            out_label, self.target_opt, self.erosion_rates, self.dilation_rates, segment_info=pos_data)
        if DEBUG:
            print('sampling and augmentation time: ', time.perf_counter() - self.start_time)
        out_weight = seg_to_weights(out_target, self.weight_opt, out_valid, out_label)
        out_volume = np.expand_dims(out_volume, 0)
        out_volume = normalize_image(out_volume, self.data_mean, self.data_std)
        # plt.imshow(out_volume[0,15,:,:]*out_target[0][1,0,15,:,:]/out_target[0][1,0,15,:,:].max()*0.5 + out_volume[0,15,:,:]*0.5)
        # plt.savefig('/braindat/lab/liusl/flywire/experiment/debug_dataset/' + str(cord) + '.png')
        return pos_data, out_volume, out_target, out_weight

    def _connector_to_volume_sample(self, connector):
        cord = [connector[2][1:-1].split(' ')[0], connector[2][1:-1].split(' ')[8], connector[2][1:-1].split(' ')[17]]
        cord = self.fafb_to_block(float(cord[0]), float(cord[1]), float(cord[2]))
        # sample image volume only
        pos, out_volume = self._crop_with_pos([0, cord[2] - self.sample_volume_size[0] / 2,
                                               cord[0] - self.sample_volume_size[1] / 2,
                                               cord[1] - self.sample_volume_size[2] / 2], self.sample_volume_size)
        out_volume = np.expand_dims(out_volume, 0)
        out_volume = normalize_image(out_volume, self.data_mean, self.data_std)

        return pos, out_volume

    def fafb_to_block(self, x, y, z):
        '''
        (x,y,z):fafb坐标
        (x_block,y_block,z_block):block块号
        (x_pixel,y_pixel,z_pixel):块内像素号,其中z为帧序号,为了使得z属于中间部分,强制z属于[29,54)
        文件名：z/y/z-xxx-y-xx-x-xx
        '''
        x_block_float = (x + 17631) / 1736 / 4
        y_block_float = (y + 19211) / 1736 / 4
        z_block_float = (z - 15) / 26
        x_block = math.floor(x_block_float)
        y_block = math.floor(y_block_float)
        z_block = math.floor(z_block_float)
        x_pixel = (x_block_float - x_block) * 1736
        y_pixel = (y_block_float - y_block) * 1736
        z_pixel = (z - 15) - z_block * 26
        while z_pixel < 28:
            z_block = z_block - 1
            z_pixel = z_pixel + 26
        return [int(x_pixel) + 156, int(y_pixel) + 156, int(z_pixel) - 12]

    def is_valid_offset(self, offset, cord):
        lb = offset + np.transpose([np.asarray([cord[2], cord[1], cord[0]])]) - np.transpose([self.sample_volume_size / 2])
        valid_lb = np.asarray([[0], [0], [0]])
        ub = offset + np.transpose([np.asarray([cord[2], cord[1], cord[0]])]) + np.transpose([self.sample_volume_size / 2])
        valid_ub = np.transpose([self.volume[0].shape])
        return np.all(lb.astype(np.int32) > valid_lb) and np.all(ub.astype(np.int32) < valid_ub)

    def sample_from_normal3d(self, sigma_z, sigma_xy, point_num):
        # print(np.asarray([np.random.normal(0, sigma_z, point_num),
        #                        np.random.normal(0, sigma_xy, point_num),
        #                        np.random.normal(0, sigma_xy, point_num)]))
        return np.asarray([np.random.normal(0, sigma_z, point_num),
                               np.random.normal(0, sigma_xy, point_num),
                               np.random.normal(0, sigma_xy, point_num)])
