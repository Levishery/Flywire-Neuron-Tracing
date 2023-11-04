from __future__ import print_function, division
from typing import Optional, List
import numpy as np
import pandas as pd
import time
import random
import os
import cv2
from cloudvolume import CloudVolume
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tf

import torch.utils.data
from ..augmentation import Compose
from ..utils import *
from skimage.transform import resize

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]
DEBUG = 0
seg2target_factor = 1


class Snemi3dDataset(torch.utils.data.Dataset):
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

    def __init__(self, sample_path, model_input_size, iter_num: int = -1, mode='train',
                 augmentor: AUGMENTOR_TYPE = None, label_name=None, sample_volume_size=None, connector_dataset=False,
                 **kwargs):

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.connector_dataset = connector_dataset
        if connector_dataset:
            self.samples = readh5(sample_path)
            self.connector_df = pd.read_csv(label_name)
            self.total_samples = len(self.connector_df)
            self.iter_num = 200 * max(
                iter_num, len(self.connector_df)) if self.mode == 'train' else len(self.connector_df)
            print('Total number of samples to be generated: ', self.iter_num)
        else:
            self.pos_samples = readh5(sample_path.split('#')[0])
            self.neg_samples = readh5(sample_path.split('#')[1])
            self.total_samples = len(self.pos_samples) + len(self.neg_samples)
            self.total_pos_samples = len(self.pos_samples)
            self.iter_num = 200 * max(
                iter_num, self.total_samples) if self.mode == 'train' else self.total_samples
            print('Total number of samples to be generated: ', self.iter_num)

        # data format
        self.sample_volume_size = np.asarray(sample_volume_size)
        self.model_input_size = model_input_size
        if isinstance(augmentor, dict):
            self.augmentor = augmentor['augmentor_before']
            self.section_augmentor = augmentor['augmentor_after']
        else:
            self.augmentor = augmentor


        # dataset: channels, depths, rows, cols
        # volume size, could be multi-volume input
        self.model_input_size = model_input_size

        # random sample factors
        self.alpha_neg = 5
        self.alpha_offset = 20
        self.neg_point_num = 300

        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training.

    def __len__(self):
        # total number of possible samples
        return self.iter_num

    def __getitem__(self, index):
        # orig input: keep uint/int format to save cpu memory
        # output sample: need np.float32
        if self.connector_dataset:
            if self.mode == 'train':
                # random sample during training
                idx = random.randint(0, len(self.connector_df) - 1)
                connector = self.connector_df[idx]
                pos_data, out_volume, out_target, out_weight = self._connector_to_target_sample(connector)
                return pos_data, out_volume, out_target, out_weight

            elif self.mode == 'val':
                connector = self.connector_df[index]
                pos_data, out_volume, out_target, out_weight = self._connector_to_target_sample_vali(connector)
                return pos_data, out_volume, out_target, out_weight

            elif self.mode == 'test':
                connector = self.connector_df[index]
                return self._connector_to_target_sample_test(connector)
        else:
            if self.mode == 'train':
                idx = random.randint(0, self.total_samples - 1)
                pos_data, out_volume, out_target, out_weight = self._get_morph_sample(idx)
                return pos_data, out_volume, out_target, out_weight
            elif self.mode == 'val':
                pos_data, out_volume, out_target, out_weight = self._get_morph_sample(index)
                return pos_data, out_volume, out_target, out_weight


    def _connector_to_target_sample(self, connector):
        samples = pd.read_csv(connector, header=None)
        # positive
        if random.random() > 0.7:
            idx = 0
        else:
            if len(samples)>1:
                idx = random.randint(1,len(samples)-1)
            else:
                idx = 0
        id0 = samples[0][idx]
        id1 = samples[1][idx]
        file_name = str(id0) + '_' + str(id1) + '.h5'
        target = samples[3][idx]
        volume = readh5(os.path.join(self.label_name, file_name))
        if random.random() < 0.33:
            volume = np.transpose(volume, [1,0,2])
        elif random.random() > 0.66:
            volume = np.transpose(volume, [2,0,1])
        # crop and resize
        z_width = int(self.sample_volume_size[0]*400/128)
        out_label = crop_volume(volume, [z_width, self.sample_volume_size[1], self.sample_volume_size[2]], [(volume.shape[0]-z_width)/2,0,0])
        out_label = resize(out_label, self.sample_volume_size, order=0, mode='constant', cval=0, clip=True, preserve_range=True,
                              anti_aliasing=False)
        data = {'image': out_label,
                'label': out_label,
                'valid_mask': out_label}
        augmented = self.augmentor(data)
        label = augmented['label']
        seg_0_morph = np.expand_dims(np.array(label == 1), 0)
        seg_1_morph = np.expand_dims(np.array(label == 2), 0)
        seg_combined_morph = np.logical_or(seg_1_morph, seg_0_morph)
        if random.random() > 0.5:
            out_volume = np.concatenate((seg_0_morph.astype(np.float32), seg_1_morph.astype(np.float32),
                                     seg_combined_morph.astype(np.float32)))
        else:
            out_volume = np.concatenate((seg_1_morph.astype(np.float32), seg_0_morph.astype(np.float32),
                                         seg_combined_morph.astype(np.float32)))
        return [0, 0, 0, 0], out_volume, [[target]], [[np.asarray([0])]]

    def _connector_to_target_sample_vali(self, connector):
        # positive
        id0 = connector[0]
        id1 = connector[1]
        file_name = str(id0) + '_' + str(id1) + '.h5'
        target = connector[3]
        volume = readh5(os.path.join(self.label_name, file_name))
        if self.test_axis == 1:
            volume = np.transpose(volume, [1,0,2])
        elif self.test_axis == 2:
            volume = np.transpose(volume, [2,0,1])
        # crop and resize
        z_width = int(self.sample_volume_size[0] * 400 / 128)
        out_label = crop_volume(volume, [z_width, self.sample_volume_size[1], self.sample_volume_size[2]],
                                [(volume.shape[0] - z_width) / 2, 0, 0])
        out_label = resize(out_label, self.sample_volume_size, order=0, mode='constant', cval=0, clip=True, preserve_range=True,
                              anti_aliasing=False)
        seg_0_morph = np.expand_dims(np.array(out_label == 1), 0)
        seg_1_morph = np.expand_dims(np.array(out_label == 2), 0)
        seg_combined_morph = np.logical_or(seg_1_morph, seg_0_morph)
        out_volume = np.concatenate((seg_0_morph.astype(np.float32), seg_1_morph.astype(np.float32),
                                     seg_combined_morph.astype(np.float32)))
        return [0, 0, 0, 0], out_volume, [[target]], [[np.asarray([0])]]

    def _connector_to_target_sample_test(self, connector):
        # positive
        id0 = connector[0]
        id1 = connector[1]
        file_name = str(id0) + '_' + str(id1) + '.h5'
        target = connector[3]
        volume = readh5(os.path.join(self.label_name, file_name))
        if self.test_axis == 1:
            volume = np.transpose(volume, [1,0,2])
        elif self.test_axis == 2:
            volume = np.transpose(volume, [2,0,1])
        # crop and resize
        z_width = int(self.sample_volume_size[0] * 400 / 128)
        out_label = crop_volume(volume, [z_width, self.sample_volume_size[1], self.sample_volume_size[2]],
                                [(volume.shape[0] - z_width) / 2, 0, 0])
        out_label = resize(out_label, self.sample_volume_size, order=0, mode='constant', cval=0, clip=True, preserve_range=True,
                              anti_aliasing=False)
        seg_0_morph = np.expand_dims(np.array(out_label == 1), 0)
        seg_1_morph = np.expand_dims(np.array(out_label == 2), 0)
        seg_combined_morph = np.logical_or(seg_1_morph, seg_0_morph)
        out_volume = np.concatenate((seg_0_morph.astype(np.float32), seg_1_morph.astype(np.float32),
                                     seg_combined_morph.astype(np.float32)))
        return [0, 0, 0, 0], out_volume, out_label, [[target]], [id0, id1]

    def _get_morph_sample(self, idx):
        if idx < self.total_pos_samples:
            out_label = self.pos_samples[idx, :, :, :]
            target = 1
        else:
            out_label = self.neg_samples[idx-self.total_pos_samples, :, :, :]
            target = 0
        seg_0_morph = np.expand_dims(np.array(out_label == 1), 0)
        seg_1_morph = np.expand_dims(np.array(out_label == 2), 0)
        seg_combined_morph = np.logical_or(seg_1_morph, seg_0_morph)
        out_volume = np.concatenate((seg_0_morph.astype(np.float32), seg_1_morph.astype(np.float32),
                                     seg_combined_morph.astype(np.float32)))
        return [0, 0, 0, 0], out_volume, [[target]], [[np.asarray([0])]]

