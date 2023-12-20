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
                 target_opt: TARGET_OPT_TYPE = ['1'],
                 weight_opt: WEIGHT_OPT_TYPE = [['1']],
                 erosion_rates: Optional[List[int]] = None,
                 dilation_rates: Optional[List[int]] = None,
                 augmentor: AUGMENTOR_TYPE = None, label_name=None, sample_volume_size=None, connector_dataset=False,
                 **kwargs):

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.connector_dataset = connector_dataset
        if connector_dataset:
            if self.mode != 'test':
                self.segmentation = readh5(sample_path)
                self.gt_labels = readh5(sample_path.replace('.h5', '-gt.h5'))
                self.images = readh5(sample_path.replace('.h5', '-img.h5'))
                self.total_samples = len(self.images)
                self.connector_df = pd.read_csv(label_name)[:self.total_samples]
            else:
                self.pos_segmentation = readh5(sample_path.split('#')[0])
                self.neg_segmentation = readh5(sample_path.split('#')[1])
                self.total_samples = len(self.pos_segmentation) + len(self.neg_segmentation)
                self.pos_images = readh5(sample_path.replace('.h5', '-img.h5').split('#')[0])
                self.neg_images = readh5(sample_path.replace('.h5', '-img.h5').split('#')[1])
                self.segmentation = np.concatenate((self.pos_segmentation, self.neg_segmentation), axis=0)
                del self.pos_segmentation, self.neg_segmentation
                self.images = np.concatenate((self.pos_images, self.neg_images), axis=0)
                del self.pos_images, self.neg_images
                self.connector_df = pd.read_csv(label_name)
                assert len(self.connector_df) == self.total_samples
            self.iter_num = 200 * max(
                iter_num, len(self.images)) if self.mode == 'train' else len(self.images)
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
        self.data_mean = 0.5
        self.data_std = 0.5

        # target and weight options
        self.target_opt = target_opt
        self.weight_opt = weight_opt
        # For 'all', users will create their own targets
        if self.target_opt[-1] == 'all':
            self.target_opt = self.target_opt[:-1]
            self.weight_opt = self.weight_opt[:-1]
        self.erosion_rates = erosion_rates
        self.dilation_rates = dilation_rates
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
                idx = random.randint(0, self.total_samples - 1)
                pos_data, out_volume, out_target, out_weight = self._connector_to_target_sample(idx)
                return pos_data, out_volume, out_target, out_weight

            elif self.mode == 'val':
                pos_data, out_volume, out_target, out_weight = self._connector_to_target_sample(index)
                return pos_data, out_volume, out_target, out_weight

            elif self.mode == 'test':
                return self._connector_to_target_sample_test(index)
        else:
            if self.mode == 'train':
                if random.random() > 0.5:
                    idx = random.randint(0, self.total_pos_samples - 1)
                    pos_data, out_volume, out_target, out_weight = self._get_morph_sample(idx)
                else:
                    idx = random.randint(self.total_pos_samples, self.total_samples - 1)
                    pos_data, out_volume, out_target, out_weight = self._get_morph_sample(idx)
                return pos_data, out_volume, out_target, out_weight
            elif self.mode == 'val':
                pos_data, out_volume, out_target, out_weight = self._get_morph_sample(index)
                return pos_data, out_volume, out_target, out_weight


    def sample_from_normal3d(self, sigma_z, sigma_xy, point_num):
        # print(np.asarray([np.random.normal(0, sigma_z, point_num),
        #                        np.random.normal(0, sigma_xy, point_num),
        #                        np.random.normal(0, sigma_xy, point_num)]))
        return np.asarray([np.random.normal(0, sigma_z, point_num),
                           np.random.normal(0, sigma_xy, point_num),
                           np.random.normal(0, sigma_xy, point_num)])

    def _connector_to_target_sample(self, idx):

        out_label = self.segmentation[idx, :, :, :]
        out_gt = self.gt_labels[idx, :, :, :]
        out_volume = self.images[idx, :, :, :].astype(np.float32)
        [seg_positive, seg_start] = [self.connector_df['label_one'][idx], self.connector_df['label_two'][idx]]
        assert seg_start in np.unique(out_label) and seg_positive in np.unique(out_label)
        seg_target_relabeled = [seg_start, seg_positive]
        neg_cord_offset = self.sample_from_normal3d(self.model_input_size[0] / self.alpha_neg,
                                                    self.model_input_size[1] / self.alpha_neg, self.neg_point_num)
        neg_cord_offset = np.asarray(
            [np.clip(neg_cord_offset[0], - self.model_input_size[0] / 2, self.model_input_size[0] / 2 - 1),
             np.clip(neg_cord_offset[1], - self.model_input_size[1] / 2, self.model_input_size[1] / 2 - 1),
             np.clip(neg_cord_offset[2], - self.model_input_size[2] / 2, self.model_input_size[2] / 2 - 1)])
        neg_cord_pos = neg_cord_offset + np.transpose(np.asarray([np.asarray(self.model_input_size) / 2]))
        neg_cord_pos = list(neg_cord_pos.astype(np.int32))

        seg_negative = np.setdiff1d(np.asarray(np.unique(out_label[neg_cord_pos[0], neg_cord_pos[1], neg_cord_pos[2]])),
                                    [0, seg_positive, seg_start])
        np.random.shuffle(seg_negative)
        # fixed number of seg_negative
        seg_negative = seg_negative[:20]

        pos_data = {'pos': [0, 0, 0, 0],
                    'seg_start': seg_start,
                    'seg_positive': seg_positive,
                    'seg_target_relabeled': seg_target_relabeled,
                    'seg_negative': seg_negative}
        out_target = seg_to_targets(
            out_label, self.target_opt, self.erosion_rates, self.dilation_rates, segment_info=pos_data)
        out_weight = seg_to_weights(out_target, self.weight_opt, None, out_label)
        out_volume = np.expand_dims(out_volume, 0)
        out_volume = normalize_image(out_volume, self.data_mean, self.data_std)

        return pos_data, out_volume, out_target, out_weight

    def _connector_to_target_sample_test(self, idx):
        out_label = self.segmentation[idx, :, :, :]
        out_volume = self.images[idx, :, :, :].astype(np.float32)
        [seg_candidate, seg_start] = [self.connector_df['label_one'][idx], self.connector_df['label_two'][idx]]
        assert seg_start in np.unique(out_label) and seg_candidate in np.unique(out_label)

        pos_data = {'pos': [0, 0, 0, 0],
                    'seg_start': seg_start,
                    'seg_candidate': seg_candidate}
        if out_volume.shape != tuple(self.model_input_size):
            out_volume = crop_volume(out_volume, self.model_input_size)
            out_label = crop_volume(out_label, self.model_input_size)
        out_volume = np.expand_dims(out_volume, 0)
        out_volume = normalize_image(out_volume, self.data_mean, self.data_std)

        return pos_data, out_volume, out_label, [[np.asarray([0])]], [[np.asarray([0])]]

    def _get_morph_sample(self, idx):
        if idx < self.total_pos_samples:
            out_label = self.pos_samples[idx, :, :, :]
            target = 1
        else:
            out_label = self.neg_samples[idx - self.total_pos_samples, :, :, :]
            target = 0
        seg_0_morph = np.expand_dims(np.array(out_label == 1), 0)
        seg_1_morph = np.expand_dims(np.array(out_label == 2), 0)
        seg_combined_morph = np.logical_or(seg_1_morph, seg_0_morph)
        out_volume = np.concatenate((seg_0_morph.astype(np.float32), seg_1_morph.astype(np.float32),
                                     seg_combined_morph.astype(np.float32)))
        return [0, 0, 0, 0], out_volume, [[target]], [[np.asarray([0])]]
