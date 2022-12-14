from __future__ import print_function, division
from typing import Optional, Tuple, List, Union
import numpy as np
from connectomics.model.loss import ConnectionLoss
import torch


def get_padsize(pad_size: Union[int, List[int]], ndim: int = 3) -> Tuple[int]:
    """Convert the padding size for 3D input volumes into numpy.pad compatible format.

    Args:
        pad_size (int, List[int]): number of values padded to the edges of each axis. 
        ndim (int): the dimension of the array to be padded. Default: 3
    """
    if type(pad_size) == int:
        pad_size = [tuple([pad_size, pad_size]) for _ in range(ndim)]
        return tuple(pad_size)

    assert len(pad_size) in [1, ndim, 2*ndim]
    if len(pad_size) == 1:
        pad_size = pad_size[0]
        pad_size = [tuple([pad_size, pad_size]) for _ in range(ndim)]
        return tuple(pad_size)

    if len(pad_size) == ndim:
        return tuple([tuple([x, x]) for x in pad_size])

    return tuple(
        [tuple([pad_size[2*i], pad_size[2*i+1]])
            for i in range(len(pad_size) // 2)])


def array_unpad(data: np.ndarray,
                pad_size: Tuple[int]) -> np.ndarray:
    """Unpad a given numpy.ndarray based on the given padding size.

    Args:
        data (numpy.ndarray): the input volume to unpad.
        pad_size (tuple): number of values removed from the edges of each axis. 
            Should be in the format of ((before_1, after_1), ... (before_N, after_N)) 
            representing the unique pad widths for each axis.
    """
    diff = data.ndim - len(pad_size)
    if diff > 0:
        extra = [(0, 0) for _ in range(diff)]
        pad_size = tuple(extra + list(pad_size))

    assert len(pad_size) == data.ndim
    index = tuple([
        slice(pad_size[i][0], data.shape[i]-pad_size[i][1])
        for i in range(data.ndim)
    ])
    return data[index]


def normalize_range(image: np.ndarray) -> np.ndarray:
    """Normalize the input image to (0,1) range and cast 
    to numpy.uint8 dtype.
    """
    eps = 1e-6
    normalized = (image - image.min()) / float(image.max() - image.min() + eps)
    normalized = (normalized*255).astype(np.uint8)
    return normalized


def normalize_z(image: np.ndarray, clip=False) -> np.ndarray:
    """Normalize the input image to (0,1) range and clip and do z score normalization. Cast
    to numpy.uint8 dtype.
    """
    eps = 1e-6
    mean = float(image.mean())
    std = float(image.std())
    if clip:
        lower_bound = np.percentile(image, 0.5)
        upper_bound = np.percentile(image, 99.5)
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean)/std
    normalized = (image - image.min()) / float(image.max() - image.min() + eps)
    normalized = (normalized*255).astype(np.uint8)
    return normalized


def normalize_image(image: np.ndarray,
                    mean: float = 0.5,
                    std: float = 0.5) -> np.ndarray:
    assert image.dtype == np.float32
    image = (image - mean) / std
    return image


def split_masks(label):
    indices = np.unique(label)
    if len(indices) > 1:
        if indices[0] == 0:
            indices = indices[1:]
        masks = [(label == x).astype(np.uint8) for x in indices]
        return np.stack(masks, 0)

    return np.ones_like(label).astype(np.uint8)[np.newaxis]


def get_connection_distance(pred, target):
    func = ConnectionLoss()
    dist_pos, dist_neg, classification = func(pred, target[0], get_distance=True)
    return dist_pos, dist_neg, classification

def get_connection_ranking(pred, ffn_label, seg_start, candidates):
    batch_size = pred.shape[0]
    for b in range(batch_size):
        pred_b = pred[b]
        ffn_label_b = ffn_label[b]
        seg_start_b = seg_start[b]
        print('ranking for segment ', seg_start_b)
        candidates_b = candidates[b]
        mask_seg_start = ffn_label_b==seg_start_b
        seg_start_embedding = pred_b[:, mask_seg_start]
        seg_start_mean_embedding = torch.mean(seg_start_embedding, dim=1)
        distance_dict = {}
        for candidate in candidates_b:
            if candidate != seg_start_b:
                mask = ffn_label_b==candidate
                condidate_embedding = pred_b[:, mask]
                condidate_mean_embedding = torch.mean(condidate_embedding, dim=1)
                distance = torch.norm(condidate_mean_embedding - seg_start_mean_embedding)
                distance_dict[candidate] = torch.norm(condidate_mean_embedding - seg_start_mean_embedding)
                # print(distance_dict[candidate])
                # print(candidate)
                # if seg_start_b == 9835396981 and distance < 2.2:
                #     print('connection to seg_start: ', candidate)
                # if distance < 2.2 and candidate == 9835396981:
                #     print('can be connected to seg_start: ', seg_start_b)
                # if seg_start_b == 8792571643 and distance < 2.2:
                #     print('connection to seg_start: ', candidate)
                # if distance < 2.2 and candidate == 8792571643:
                #     print('can be connected to seg_start: ', seg_start_b)
        distance_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])}
        # if seg_start_b == 9835396981 or seg_start_b == 8792571643:
        print('distance_dict: ', distance_dict)
