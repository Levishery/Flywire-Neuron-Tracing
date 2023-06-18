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


class PatchDataset(torch.utils.data.Dataset):
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

    def __init__(self, block_path, model_input_size, data_mean, data_std, mode='test'):

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.model_input_size = model_input_size
        self.data_mean = data_mean
        self.data_std = data_std

        # dataset: channels, depths, rows, cols
        # volume size, could be multi-volume input
        self.model_input_size = model_input_size
        self.data_fid = h5py.File(block_path, 'r')
        self.patch_names = self.data_fid.keys()

        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training.
        if self.mode == 'test':
            self.iter_num = len(self.patch_names)
        else:
            raise ValueError('Patch dataset is for test only')
        print('Total number of samples to be generated: ', self.iter_num)

    def __len__(self):
        # total number of possible samples
        return self.iter_num

    def __getitem__(self, index):
        # orig input: keep uint/int format to save cpu memory
        # output sample: need np.float32
        if self.mode == 'test':
            return self.sample_test(index)

    def sample_test(self, index):
        patch = np.transpose(np.array(self.data_fid[list(self.patch_names)[index]]), [2, 1, 0])
        out_volume = (patch / 255.0).astype(np.float32)
        out_volume = normalize_image(out_volume, self.data_mean, self.data_std)
        return list(self.patch_names)[index], out_volume, 0, 0, 0
