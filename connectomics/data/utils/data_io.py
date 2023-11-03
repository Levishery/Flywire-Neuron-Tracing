from __future__ import print_function, division

import pickle
from typing import Optional, List

import os
import h5py
import math
import glob
import numpy as np
import imageio
from scipy.ndimage import zoom
import pandas as pd
from tqdm import tqdm


def fafb_to_block(x, y, z, return_pixel=False):
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
    if return_pixel:
        return x_block, y_block, z_block, np.round(x_pixel), np.round(y_pixel), np.round(z_pixel)
    else:
        return x_block, y_block, z_block


def readimg_as_vol(filename, drop_channel=True):
    img_suf = filename[filename.rfind('.') + 1:]
    assert img_suf in ['png', 'tif']
    data = imageio.imread(filename)
    if drop_channel and data.ndim == 3:
        # convert RGB image to grayscale by average
        data = np.mean(data, axis=-1).astype(np.uint8)
    return data[np.newaxis, :, :]


def readh5(filename, dataset=None):
    fid = h5py.File(filename, 'r')
    if dataset is None:
        dataset = list(fid)[0]
    return np.array(fid[dataset])


def stat_biological_recall(potential_path):
    # potential_path = '/braindat/lab/liusl/flywire/block_data/neuroglancer/connector_22_7_97-101.csv'
    names = potential_path.split('.')[0].split('/')[-1]
    name_prefix = names.split('_')[:-1]
    name_prefix = name_prefix[0] + '_' + name_prefix[1] + '_' + name_prefix[2] + '_'
    z_start = int(names.split('_')[-1].split('-')[0])
    z_end = int(names.split('_')[-1].split('-')[1])
    block_paths = [
        '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/%s.csv' % (name_prefix + str(z_start)),
        '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/%s.csv' % (
                name_prefix + str(z_start + 1)),
        '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/%s.csv' % (
                name_prefix + str(z_start + 2)),
        '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/%s.csv' % (
                name_prefix + str(z_start + 3)),
        '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/%s.csv' % (
                name_prefix + str(z_start + 4))]

    edges = pd.read_csv(potential_path, header=None)
    total_positives = 0
    hit = 0
    for block_path in block_paths:
        if os.path.exists(block_path):
            samples = pd.read_csv(block_path, header=None)
            positives_indexes = np.where(samples[3] == 1)
            query = list(samples[0][list(positives_indexes[0])])
            pos = list(samples[1][list(positives_indexes[0])])
            for i in range(len(query)):
                total_positives = total_positives + 1
                potentials = list(edges[1][np.where(edges[0] == query[i])[0]])
                if pos[i] in potentials:
                    hit = hit + 1
                    continue
                potentials = list(edges[1][np.where(edges[0] == pos[i])[0]])
                if query[i] in potentials:
                    hit = hit + 1
                    continue
    print('bilogical edge recall: ', hit / total_positives)


def readvol(filename, dataset=None):
    r"""Load a image volume in HDF5, TIFF or PNG formats.
    """
    img_suf = filename[filename.rfind('.') + 1:]
    if img_suf in ['h5', 'hdf5']:
        data = readh5(filename, dataset)
    elif 'tif' in img_suf:
        data = imageio.volread(filename).squeeze()
    elif 'png' in img_suf:
        data = readimgs(filename)
    else:
        raise ValueError('unrecognizable file format for %s' % (filename))
    return data


def savevol(filename, vol, dataset='main', format='h5'):
    if format == 'h5':
        writeh5(filename, vol, dataset='main')
    if format == 'png':
        currentDirectory = os.getcwd()
        img_save_path = os.path.join(currentDirectory, filename)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        for i in range(vol.shape[0]):
            imageio.imsave('%s/%04d.png' % (img_save_path, i), vol[i])


def readim(filename, do_channel=False):
    # x,y,c
    if not os.path.exists(filename):
        im = None
    else:  # note: cv2 do "bgr" channel order
        im = imageio.imread(filename)
        if do_channel and im.ndim == 2:
            im = im[:, :, None]
    return im


def readimgs(filename):
    filelist = sorted(glob.glob(filename))
    num_imgs = len(filelist)

    # decide numpy array shape:
    img = imageio.imread(filelist[0])
    data = np.zeros((num_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)
    data[0] = img

    # load all images
    if num_imgs > 1:
        for i in range(1, num_imgs):
            data[i] = imageio.imread(filelist[i])

    return data


def writeh5(filename, dtarray, dataset='main'):
    fid = h5py.File(filename, 'w')
    if isinstance(dataset, (list,)):
        for i, dd in enumerate(dataset):
            ds = fid.create_dataset(
                dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(dataset, dtarray.shape,
                                compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()


def create_json(ndim: int = 1, dtype: str = "uint8", data_path: str = "/path/to/data/",
                height: int = 10000, width: int = 10000, depth: int = 500,
                n_columns: int = 3, n_rows: int = 3, tile_size: int = 4096,
                tile_ratio: int = 1, tile_st: List[int] = [0, 0]):
    """Create a dictionay to store the metadata for large volumes. The dictionary is
    usually saved as a JSON file and can be read by the TileDataset.

    Args:
        ndim (int, optional): [description]. Defaults to 1.
        dtype (str, optional): [description]. Defaults to "uint8".
        data_pathLstr (str, optional): [description]. Defaults to "/path/to/data".
        height (int, optional): [description]. Defaults to 10000.
        width (int, optional): [description]. Defaults to 10000.
        depth (int, optional): [description]. Defaults to 500.
        tile_ratio (int, optional): [description]. Defaults to 1.
        n_columns (int, optional): [description]. Defaults to 3.
        n_rows (int, optional): [description]. Defaults to 3.
        tile_size (int, optional): [description]. Defaults to 4096.
        tile_ratio (int, optional): [description]. Defaults to 1.
        tile_st (List[int], optional): [description]. Defaults to [0,0].
    """
    metadata = {}
    metadata["ndim"] = ndim
    metadata["dtype"] = dtype

    digits = int(math.log10(depth)) + 1
    metadata["image"] = [
        data_path + str(i).zfill(digits) + r"/{row}_{column}.png"
        for i in range(depth)]

    metadata["height"] = height
    metadata["width"] = width
    metadata["depth"] = depth

    metadata["n_columns"] = n_columns
    metadata["n_rows"] = n_rows

    metadata["tile_size"] = tile_size
    metadata["tile_ratio"] = tile_ratio
    metadata["tile_st"] = tile_st

    return metadata


####################################################################
# tile to volume
####################################################################


def vast2Seg(seg):
    # convert to 24 bits
    if seg.ndim == 2 or seg.shape[-1] == 1:
        return np.squeeze(seg)
    elif seg.ndim == 3:  # 1 rgb image
        return seg[:, :, 0].astype(np.uint32) * 65536 + seg[:, :, 1].astype(np.uint32) * 256 + seg[:, :, 2].astype(
            np.uint32)
    elif seg.ndim == 4:  # n rgb image
        return seg[:, :, :, 0].astype(np.uint32) * 65536 + seg[:, :, :, 1].astype(np.uint32) * 256 + seg[:, :, :,
                                                                                                     2].astype(
            np.uint32)


def tile2volume(tiles: List[str], coord: List[int], coord_m: List[int], tile_sz: int,
                dt: type = np.uint8, tile_st: List[int] = [0, 0], tile_ratio: float = 1.0,
                do_im: bool = True, background: int = 128) -> np.ndarray:
    """Construct a volume from image tiles based on the given volume coordinate.

    Args:
        tiles (List[str]): a list of paths to the image tiles.
        coord (List[int]): the coordinate of the volume to be constructed.
        coord_m (List[int]): the coordinate of the whole dataset with the tiles.
        tile_sz (int): the height and width of the tiles, which is assumed to be square.
        dt (type): data type of the constructed volume. Default: numpy.uint8
        tile_st (List[int]): start position of the tiles. Default: [0, 0]
        tile_ratio (float): scale factor for resizing the tiles. Default: 1.0
        do_im (bool): construct an image volume (apply linear interpolation for resizing). Default: `True`
        background (int): background value for filling the constructed volume. Default: 128
    """
    z0o, z1o, y0o, y1o, x0o, x1o = coord  # region to crop
    z0m, z1m, y0m, y1m, x0m, x1m = coord_m  # tile boundary

    bd = [max(-z0o, z0m), max(0, z1o - z1m), max(-y0o, y0m),
          max(0, y1o - y1m), max(-x0o, x0m), max(0, x1o - x1m)]
    z0, y0, x0 = max(z0o, z0m), max(y0o, y0m), max(x0o, x0m)
    z1, y1, x1 = min(z1o, z1m), min(y1o, y1m), min(x1o, x1m)

    result = background * np.ones((z1 - z0, y1 - y0, x1 - x0), dt)
    c0 = x0 // tile_sz  # floor
    c1 = (x1 + tile_sz - 1) // tile_sz  # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz - 1) // tile_sz
    for z in range(z0, z1):
        pattern = tiles[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                if r'{row}_{column}' in pattern:
                    path = pattern.format(
                        row=row + tile_st[0], column=column + tile_st[1])
                else:
                    path = pattern
                patch = readim(path, do_channel=True)
                if patch is not None:
                    if tile_ratio != 1:  # im ->1, label->0
                        patch = zoom(
                            patch, [tile_ratio, tile_ratio, 1], order=int(do_im))

                    # last tile may not be of the same size
                    xp0 = column * tile_sz
                    xp1 = xp0 + patch.shape[1]
                    yp0 = row * tile_sz
                    yp1 = yp0 + patch.shape[0]
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    if do_im:  # image
                        result[z - z0, y0a - y0:y1a - y0, x0a - x0:x1a -
                                                                   x0] = patch[y0a - yp0:y1a - yp0, x0a - xp0:x1a - xp0,
                                                                         0]
                    else:  # label
                        result[z - z0, y0a - y0:y1a - y0, x0a - x0:x1a -
                                                                   x0] = vast2Seg(
                            patch[y0a - yp0:y1a - yp0, x0a - xp0:x1a - xp0])

    # For chunks touching the border of the large input volume, apply padding.
    if max(bd) > 0:
        result = np.pad(
            result, ((bd[0], bd[1]), (bd[2], bd[3]), (bd[4], bd[5])), 'reflect')
    return result


def get_blocknames_from_points(input_file_path):
    if not os.path.exists(input_file_path):
        os.makedirs(input_file_path, exist_ok=True)
        with open(input_file_path + '.pkl', "rb") as f:
            seq_data = pickle.load(f)
        for j in tqdm(range(len(seq_data))):
            seq = seq_data[j]
            points = seq[1]
            for i in range(points.shape[0]):
                point = points[i, :]
                candidates = list(seq[0])
                x_block, y_block, z_block = fafb_to_block(point[0], point[1], point[2])
                block_name = 'connector_' + str(x_block) + '_' + str(y_block) + '_' + str(z_block)
                row = pd.DataFrame(
                    [{'node0_segid': int(j), 'node1_segid': str(candidates), 'cord': point, 'target': int(i),
                      'prediction': -1}])
                row.to_csv(os.path.join(input_file_path, block_name + '.csv'), mode='a', header=False, index=False)
    else:
        with open(input_file_path + '.pkl', "rb") as f:
            seq_data = pickle.load(f)
    return seq_data


def save_feature(seq_data, path):
    file_path = os.path.join(path, 'sequence_with_imagefeature.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(seq_data, file)
