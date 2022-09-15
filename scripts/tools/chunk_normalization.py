import h5py
import numpy as np
import cv2
import os
from PIL import Image
import tifffile as tf
import matplotlib.pyplot as plt

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

dir = '/braindat/lab/liusl/connectomic_dataset/H01_4*4*30'
files = os.listdir(dir)
for f in files:
    f_path = os.path.join(dir, f)
    hdf = h5py.File(f_path, 'r+')
    x = np.asarray(hdf['main']).astype(np.uint8)
    shape = x.shape
    [w,h] = [shape[1], shape[2]]
    angles = [[0,0], [0,0.99], [0.99,0], [0.99,0.99], [0,0.5], [0.5,0], [0.99, 0.5], [0.5, 0.99]]
    for i in range(shape[0]):
        slice = x[i,:,:]
        for a in angles:
            p = [int(np.floor(min(w,h)*a[0])), int(np.floor(min(w,h)*a[1]))]
            if np.mean(slice[p[0]:p[0]+5, p[1]:p[1]+5]) < 2:
                mask = np.zeros([w+2, h+2], np.uint8)
                gray = np.mean(x[i-1,:,:]) if i>0 else np.mean(x[i+1,:,:])
                slice = cv2.floodFill(slice, mask, p, gray, 10, 10, cv2.FLOODFILL_FIXED_RANGE)[1]
                x[i, :, :] = slice
    x = normalize_z(x, clip=False)
    del hdf['main']
    hdf.create_dataset('main', data=x)
    hdf.close()
    print('normalized ', f_path)
    # tf.imsave('/braindat/lab/liusl/connectomic_dataset/test_chunk/A+.tiff', x)
