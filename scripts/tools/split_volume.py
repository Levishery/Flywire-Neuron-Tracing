import h5py
import numpy as np
import os
from PIL import Image
import tifffile as tf
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def index_to_location(index, sz):
    # index -> z,y,x
    # sz: [y*x, x]
    pos = [0, 0, 0]
    pos[0] = int(np.floor(index / sz[0]))
    pz_r = index % sz[0]
    pos[1] = int(np.floor(pz_r / sz[1]))
    pos[2] = int(pz_r % sz[1])
    return pos


p = '/braindat/data_labeled/MitoEM_data/rat/im'
target_path = '/braindat/lab/liusl/connectomic_dataset/Rat_4*4*30'
start = 400
n = 100
name_ = 'rat'+str(start)+'_'
HDF = False
TIFF = False
isotropic = False
slice_skip = 1
rescale = 2
chunk_shape = [100, 1024, 1024]

if HDF:
    hdf = h5py.File(p, 'r')
    list(hdf.keys())
    x = hdf['volumes']
    list(x.keys())
    # raw = np.asarray(x['image'])
    raw = np.asarray(x['raw'])
    # label = x['labels']
    # label = np.asarray(label['neuron_ids'])

else:
    slices = os.listdir(p)
    slices.sort()
    slice = np.asarray(Image.open(os.path.join(p, slices[start])))
    raw = np.zeros([n, np.asarray(slice.shape)[0], np.asarray(slice.shape)[1]])
    raw[0, :, :] = slice
    i = 1
    for slice in slices[start+1:start+n]:
        slice = np.asarray(Image.open(os.path.join(p, slice)))
        raw[i, :, :] = slice
        i = i+1
# shape = (2, 3, 3, 100, 1024, 1024)
# chunks = np.lib.stride_tricks.as_strided(raw, strides=shape, shape=shape).reshape((-1, shape[0], shape[1], shape[2]))

# plt.imshow(raw[0, :, :])
# plt.show()
# pinky103 is too small
# chunk_shape = [72, 1280, 1280]
if isotropic:
    axis = 3
else:
    axis = 1
# raw_copy = raw.copy()
raw_copy = raw[:, 0:2048, 0:2048].copy()
for i in range(axis):
    raw = zoom(raw_copy, (1, rescale, rescale), order=1)
    name = name_ + 'axis' + str(i)
    shape = raw.shape
    split_shape = [np.floor(shape[1] / chunk_shape[1] * np.floor(shape[2] / chunk_shape[2])),
                   np.floor(shape[2] / chunk_shape[2])]
    num_chunks = int(np.floor(split_shape[0] * np.floor(shape[0] / chunk_shape[0])))
    for i in range(num_chunks):
        pos = index_to_location(i, split_shape)
        pos_voxel = [int(pos[i] * chunk_shape[i]) for i in range(3)]
        chunk = raw[pos_voxel[0]:pos_voxel[0] + chunk_shape[0]:slice_skip,
                pos_voxel[1]:pos_voxel[1] + chunk_shape[1], pos_voxel[2]:pos_voxel[2] + chunk_shape[2]]
        if TIFF:
            chunk_file = name + '_chunk_' + str(pos_voxel[0]) + '_' + str(pos_voxel[1]) + '_' + str(pos_voxel[2]) + '.tiff'
            tf.imsave(os.path.join(target_path, chunk_file), chunk)
        else:
            chunk_file = name + '_chunk_' + str(pos_voxel[0]) + '_' + str(pos_voxel[1]) + '_' + str(pos_voxel[2]) + '.h5'
            hf = h5py.File(os.path.join(target_path, chunk_file), 'w')
            hf.create_dataset('main', data=chunk.astype(np.uint8))
            hf.close()
        print('save chunk', chunk_file)
    raw_copy = raw_copy.transpose(1, 2, 0)


