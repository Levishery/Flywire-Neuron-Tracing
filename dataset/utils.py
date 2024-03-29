import numpy as np
import math
import pandas as pd
import os
import shutil
from tqdm import tqdm
import tifffile as tf
import random
import imageio
import glob
import networkx as nx
from queue import Queue,LifoQueue,PriorityQueue
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
import navis
import h5py
import re
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
from cloudvolume import CloudVolume
from skimage.transform import resize
import pickle

z_block = 272
y_block = 22
x_block = 41


def readh5(filename, dataset=None):
    fid = h5py.File(filename, 'r')
    if dataset is None:
        dataset = list(fid)[0]
    return np.array(fid[dataset])


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


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def xpblock_to_fafb(z_block, y_block, x_block, z_coo=0, y_coo=0, x_coo=0):
    '''
    函数功能:xp集群上的点到fafb原始坐标的转换
    输入:xp集群点属于的块儿和块内坐标,每个块的大小为1736*1736*26(x,y,z顺序)
    输出:fafb点坐标
    z_block:xp集群点做所在块儿的z轴序号
    y_block:xp集群点做所在块儿的y轴序号
    x_clock:xp集群点做所在块儿的x轴序号
    z_coo:xp集群点所在块内的z轴坐标 (29-54)
    y_coo:xp集群点所在块内的y轴坐标 (0-1735)
    x_coo:xp集群点所在块内的x轴坐标 (0-1735)

    '''

    # 第二个：原来的,z准确的,x,y误差超过5,block大小为 26*1736*1736
    # x_fafb = 1736 * 4 * x_block - 17830 + 4 * x_coo
    x_fafb = 1736 * 4 * x_block - 17631 + 4 * x_coo
    # y_fafb = 1736 * 4 * y_block - 19419 + 4 * y_coo
    y_fafb = 1736 * 4 * y_block - 19211 + 4 * y_coo
    z_fafb = 26 * z_block + 15 + z_coo
    # z_fafb = 26 * z_block + 30 + z_coo
    return (x_fafb, y_fafb, z_fafb)


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


def define_block_list():
    block_list = [[[]] * x_block for i in range(y_block)]
    block_list = [block_list for i in range(z_block)]
    return block_list


def split_train_test_block():
    val_num = 200
    source_dir = '/braindat/lab/liusl/flywire/block_data/2_percent'
    df = pd.read_csv('/braindat/lab/liusl/flywire/block_data/2_percent.csv')
    train_dir = '/braindat/lab/liusl/flywire/block_data/train_2_all'
    val_dir = '/braindat/lab/liusl/flywire/block_data/val_2_all'

    # total_num = 4000
    total_num = len(df)
    rows = list(range(0, total_num))
    random.shuffle(rows)
    for i in rows[0:200]:
        file_name = df['block'][i]
        shutil.copy(os.path.join(source_dir, file_name), val_dir)

    for i in rows[200:]:
        file_name = df['block'][i]
        shutil.copy(os.path.join(source_dir, file_name), train_dir)
    source_dir = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    df = pd.read_csv('/braindat/lab/liusl/flywire/block_data/v2/30_percent.csv')
    train_dir = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train'

    total_num = 4000
    for i in range(total_num):
        file_name = df['block'][i]
        shutil.copy(os.path.join(source_dir, file_name), train_dir)


def sample_by_block():
    target_tree_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data'
    target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    visualization_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/visualization'
    connector_files = os.listdir(target_connector_path)
    block_connector = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    thresh = 10

    for connector_f in tqdm(connector_files):
        # block_list = define_block_list()
        visualization_f = connector_f.replace('_connector.csv', '_save.csv')
        neuron_id = visualization_f[:-9]
        df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
        df_weight = pd.read_csv(os.path.join(visualization_path, visualization_f), index_col=0)
        if len(df) > 10:
            tree_f = connector_f.replace('_connector.csv', '.json')
            tree = navis.read_json(os.path.join(target_tree_path, tree_f))[0]
            max_strahler_index = tree.nodes.strahler_index.max()

            for i in df.index:
                weight1 = df_weight[str(int(df['node1_segid'][i]))][0]
                weight0 = df_weight[str(int(df['node0_segid'][i]))][0]
                score1 = 1 / ((1 / min(weight1, 40)) * (1 / min(weight0, 40)))
                score2 = (1 / (max_strahler_index - int(df['Strahler order'][i].split('\n')[0].split(' ')[-1]) + 3))
                score = score2 * score1
                if score > thresh:
                    cord0 = np.array(df['node0_cord'][i][1:-1].split(', ')).astype(float)
                    cord1 = np.array(df['node1_cord'][i][1:-1].split(', ')).astype(float)
                    cord = (cord0 + cord1) / 2
                    x_b, y_b, z_b = fafb_to_block(cord[0], cord[1], cord[2])
                    file_name = 'connector_' + str(x_b) + '_' + str(y_b) + '_' + str(z_b) + '.csv'
                    # node 0 is seg_start, which is a larger segment.
                    if weight0 > weight1:
                        row = pd.DataFrame(
                            [{'node0_segid': int(df['node0_segid'][i]), 'node1_segid': int(df['node1_segid'][i]),
                              'cord': cord, 'cord0': cord0, 'cord1': cord1, 'score': score, 'neuron_id': neuron_id}])
                    else:
                        row = pd.DataFrame(
                            [{'node0_segid': int(df['node1_segid'][i]), 'node1_segid': int(df['node0_segid'][i]),
                              'cord': cord, 'cord0': cord1, 'cord1': cord0, 'score': score, 'neuron_id': neuron_id}])
                    row.to_csv(os.path.join(block_connector, file_name), mode='a', header=False, index=False)
        else:
            print('invalid neuron', neuron_id)
            # block_list[z_b][y_b][x_b].append([int(df['node0_segid'][i]), int(df['node1_segid'][i]),
            #                                                    cord, score, neuron_id])
            # for i in range(x_block):
            #     for j in range(y_block):
            #         for k in range(z_block):
            #             if len(block_list[k][j][i]) > 0:
            #                 file_name = 'connector_' + str(i) + '_' + str(j) + '_' + str(k) + '.csv'
            #                 df = pd.DataFrame(block_list[k][j][i], columns=['node0_segid', 'node1_segid', 'cord',
            #                                                                 'score', 'neuron_id'])
            #                 df.to_csv(file_name, mode='a', header=True, index=False)


def stat_grow():
    block_connector = '/braindat/lab/liusl/flywire/test-skel/PointDETR/grow_vector'
    connector_files = os.listdir(block_connector)
    visualization_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/visualization'
    grow_count = 0
    segment_count = 0
    for connector_f in tqdm(connector_files):
        df = pd.read_csv(os.path.join(block_connector, connector_f),index_col=False, header=None)
        grow_count = grow_count + len(df)
        segment_count = segment_count + len(np.unique(df[0]))
    print(grow_count/segment_count)


def sample_by_neuron():
    target_tree_path = '/braindat/lab/liusl/flywire/test-skel/tree_data'
    target_connector_path = '/braindat/lab/liusl/flywire/test-skel/connector_data'
    visualization_path = '/braindat/lab/liusl/flywire/test-skel/visualization'
    connector_files = os.listdir(target_connector_path)
    block_connector = '/braindat/lab/liusl/flywire/test-skel/30_percent'
    thresh = 10

    for connector_f in tqdm(connector_files):
        # block_list = define_block_list()
        visualization_f = connector_f.replace('_connector.csv', '_save.csv')
        neuron_id = visualization_f[:-9]
        print('processing neuron', neuron_id)
        df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
        df_weight = pd.read_csv(os.path.join(visualization_path, visualization_f), index_col=0)
        if len(df) > 10:
            tree_f = connector_f.replace('_connector.csv', '.json')
            tree = navis.read_json(os.path.join(target_tree_path, tree_f))[0]
            max_strahler_index = tree.nodes.strahler_index.max()

            for i in df.index:
                weight1 = df_weight[str(int(df['node1_segid'][i]))][0]
                weight0 = df_weight[str(int(df['node0_segid'][i]))][0]
                score1 = 1 / ((1 / min(weight1, 40)) * (1 / min(weight0, 40)))
                score2 = (1 / (max_strahler_index - df['Strahler order'][i] + 3))
                score = score2 * score1
                if score > thresh:
                    cord0 = np.array(df['node0_cord'][i][1:-1].split(', ')).astype(float)
                    cord1 = np.array(df['node1_cord'][i][1:-1].split(', ')).astype(float)
                    cord = (cord0 + cord1) / 2
                    file_name = str(neuron_id) + '.csv'
                    # node 0 is seg_start, which is a larger segment.
                    if weight0 > weight1:
                        row = pd.DataFrame(
                            [{'node0_segid': int(df['node0_segid'][i]), 'node1_segid': int(df['node1_segid'][i]),
                              'cord': cord, 'cord0': cord0, 'cord1': cord1, 'score': score, 'neuron_id': neuron_id}])
                    else:
                        row = pd.DataFrame(
                            [{'node0_segid': int(df['node1_segid'][i]), 'node1_segid': int(df['node0_segid'][i]),
                              'cord': cord, 'cord0': cord1, 'cord1': cord0, 'score': score, 'neuron_id': neuron_id}])
                    row.to_csv(os.path.join(block_connector, file_name), mode='a', header=False, index=False)
        else:
            print('invalid neuron', neuron_id)
            # block_list[z_b][y_b][x_b].append([int(df['node0_segid'][i]), int(df['node1_segid'][i]),
            #                                                    cord, score, neuron_id])
            # for i in range(x_block):
            #     for j in range(y_block):
            #         for k in range(z_block):
            #             if len(block_list[k][j][i]) > 0:
            #                 file_name = 'connector_' + str(i) + '_' + str(j) + '_' + str(k) + '.csv'
            #                 df = pd.DataFrame(block_list[k][j][i], columns=['node0_segid', 'node1_segid', 'cord',
            #                                                                 'score', 'neuron_id'])
            #                 df.to_csv(file_name, mode='a', header=True, index=False)


def find_hard_blocks():
    csv_path = '/braindat/lab/liusl/flywire/block_data/v2/morph_performence.csv'
    from_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_4000'
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_hard'
    df = pd.read_csv(csv_path, index_col=0, header=None)

    for i in range(len(df)):
        recall = df.iloc[i].values[1]
        name = df.iloc[i].name
        if recall < 0.72:
            shutil.copy(os.path.join(from_path, name), os.path.join(target_path, name))


def select_hard():
    block_connector = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_4000'
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train_1000'
    target_hard_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train_1000_hard'
    target_test_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000'
    result_name = '/braindat/lab/liusl/flywire/block_data/v2/morph_performence_1k.csv'
    df = pd.read_csv(result_name, index_col=0, header=None)
    recall_list = []
    for i in range(len(df)):
        f_name = df.iloc[i].name
        # shutil.copy(os.path.join(block_connector, f_name), os.path.join(target_path, f_name))
        recall = df.iloc[i].values[1]
        recall_list.append(recall)
    recall_list.sort()
    # plt.hist(recall_list)
    # plt.show()
    thresh = recall_list[int(len(df) * 0.15)]
    name_list = []
    for i in range(len(df)):
        f_name = df.iloc[i].name
        name_list.append(f_name)
        recall = df.iloc[i].values[1]
        # if recall < thresh:
        #     shutil.copy(os.path.join(block_connector, f_name), os.path.join(target_hard_path, f_name))
    connector_list = os.listdir(block_connector)
    for f in connector_list:
        if f in name_list:
            continue
        shutil.copy(os.path.join(block_connector, f), os.path.join(target_test_path, f))


def get_neuron_list():
    proofread_status = '/braindat/lab/liusl/flywire/proof_stat_df.csv'
    used_neurons_file = '/braindat/lab/liusl/flywire/used_neurons.csv'
    target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    connector_files = os.listdir(target_connector_path)
    connector_files = [int(x.split('_')[0]) for x in connector_files]
    proof_list = pd.read_csv(proofread_status)
    used_neurons = pd.DataFrame(columns=proof_list.columns)
    frames = []
    record = []
    for i in tqdm(range(len(proof_list))):
        neuron_id = int(proof_list['pt_root_id'][i])
        if neuron_id in connector_files and neuron_id not in record:
            row = proof_list.iloc[i]
            frames.append(row)
            record.append(neuron_id)
    used_neurons = pd.concat(frames, axis=1).transpose()
    used_neurons.to_csv(used_neurons_file)


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def reformate_image_data(process_id=0):
    ### input dir: 84 png of 2048*2048
    ### output dir: a h5 dataset contains the patches whose starting cord is located in the block,
    # indexed by the cord in fafb cord
    patch_size = np.asarray([129, 129, 17])
    source_size = np.asarray([2048, 2048, 84]).astype(np.int)
    target_size = np.asarray([1736, 1736, 26]).astype(np.int)
    start_offset = ((source_size - target_size) / 2).astype(np.int)
    png_path = '/h3cstore_nt/fafbv14_png/fafbv14'
    target_path = '/h3cstore_nt/fafbv14_h5'
    connector_files = chunks(os.listdir(png_path), 4)[process_id]

    for img_dir in tqdm(connector_files):
        # indexing
        [block_x, block_y, block_z] = img_dir.split('_')[1:]
        block_x = int(block_x)
        block_y = int(block_y)
        block_z = int(block_z.split('.')[0])
        z_path = os.path.join(target_path, 'z_' + str(block_z))
        if os.path.exists(os.path.join(z_path, img_dir + '.h5')):
            continue
        print('reformate block %s'%img_dir)
        filelist = sorted(glob.glob(os.path.join(png_path, img_dir + '/*.png')))
        start_cord = np.asarray(xpblock_to_fafb(block_z, block_y, block_x, 29, 0, 0))
        end_cord = np.asarray(xpblock_to_fafb(block_z + 1, block_y + 1, block_x + 1, 29, 0, 0)) - np.asarray([1, 1, 1])
        patch_offset_list, patch_cord_list = index_patch(start_cord, end_cord)

        # read img
        assert len(filelist) == 84, "%s image missing" % img_dir
        num_imgs = len(filelist)
        img = imageio.imread(filelist[0])
        data = np.zeros((num_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)
        data[0] = img
        # load all images
        if num_imgs > 1:
            for i in range(1, num_imgs):
                try:
                    data[i] = imageio.imread(filelist[i])
                except:
                    print("%s image corrupted" % img_dir)
        data = np.transpose(data, [1, 2, 0])
        data = data[start_offset[0]:, start_offset[0]:, :]

        # write h5 dataset
        os.makedirs(z_path, exist_ok=True)
        fid = h5py.File(os.path.join(z_path, img_dir + '.h5'), 'w')
        for patch_cord, patch_offset in zip(patch_cord_list, patch_offset_list):
            _x, _y, _z, patch_offset_x, patch_offset_y, patch_offset_z = fafb_to_block(patch_cord[0], patch_cord[1],
                                                                                       patch_cord[2], return_pixel=True)
            patch = data[int(patch_offset_x):int(patch_offset_x) + patch_size[0],
                    int(patch_offset_y):int(patch_offset_y) + patch_size[1],
                    int(patch_offset_z):int(patch_offset_z) + patch_size[2]]
            name = str(int(patch_cord[0])) + ',' + str(int(patch_cord[1])) + ',' + str(int(patch_cord[2]))
            ds = fid.create_dataset(name, patch.shape, compression="gzip", dtype=patch.dtype)
            ds[:] = patch

        fid.close()


def index_patch(start_cord, end_cord):
    resolution = np.asarray([4, 4, 1])
    stride_size = np.asarray([64, 64, 8]) * resolution
    start_index = start_cord / stride_size
    end_index = end_cord / stride_size
    patch_offset_list = []
    patch_cord_list = []

    x = np.linspace(np.ceil(start_index[0]), np.floor(end_index[0]),
                    int(np.floor(end_index[0]) - np.ceil(start_index[0])) + 1)
    y = np.linspace(np.ceil(start_index[1]), np.floor(end_index[1]),
                    int(np.floor(end_index[1]) - np.ceil(start_index[1])) + 1)
    z = np.linspace(np.ceil(start_index[2]), np.floor(end_index[2]),
                    int(np.floor(end_index[2]) - np.ceil(start_index[2])) + 1)
    xv, yv, zv = np.meshgrid(x, y, z)

    for ix, iy, iz in np.ndindex(xv.shape):
        patch_cord_list.append(np.asarray([xv[ix, iy, iz], yv[ix, iy, iz], zv[ix, iy, iz]]) * stride_size)
        assert patch_cord_list[-1][0] >= start_cord[0] and patch_cord_list[-1][0] <= end_cord[0]
        assert patch_cord_list[-1][1] >= start_cord[1] and patch_cord_list[-1][1] <= end_cord[1]
        assert patch_cord_list[-1][2] >= start_cord[2] and patch_cord_list[-1][2] <= end_cord[2]
        patch_offset = (patch_cord_list[-1] - start_cord) / resolution
        patch_offset_list.append(patch_offset.astype(np.int32))
    return patch_offset_list, patch_cord_list


def stat_candidate():
    # id = 5534971521
    id = 6719166843
    # id =
    vol_ffn1 = CloudVolume('file:///h3cstore_nt/fafb-ffn1', cache=True)  #
    vol_ffn1.parallel = 8
    vol_ffn1.meta.info['skeletons'] = 'skeletons_32nm'
    vol_ffn1.skeleton.meta.refresh_info()
    vol_ffn1.skeleton.meta.info['sharding']['hash'] = 'murmurhash3_x86_128'
    vol_ffn1.skeleton = ShardedPrecomputedSkeletonSource(vol_ffn1.skeleton.meta, vol_ffn1.cache, vol_ffn1.config)

    candidates = np.asarray([])
    skel= vol_ffn1.skeleton.get(id)
    for point in tqdm(skel.vertices):
        candidates = np.append(candidates, bio_constrained_candidates(vol_ffn1, point, radius=500, move_back=0, use_direct=False))
    candidates = np.unique(candidates)
    print(len(candidates))
    print(skel.cable_length())


def bio_constrained_candidates(vol_ffn1, end_point, skel_vector=np.asarray([0,0,0]), radius=500, theta=30, move_back=100, use_direct=True):
    ### a simple implementation of biological constrained candidate finding
    end_point = end_point//[16,16,40]
    block_shape = np.asarray([2 * (radius // 16), 2 * (radius // 16), 2 * (radius // 40)])
    ffn_volume = vol_ffn1[end_point[0]-block_shape[0]//2:end_point[0]+block_shape[0]//2 + 1, end_point[1]-block_shape[1]//2:end_point[1]+block_shape[1]//2 + 1, end_point[2]-block_shape[2]//2:end_point[2]+block_shape[2]//2 + 1].squeeze()
    cos_theta = math.cos(theta/360*2*math.pi)
    mask = np.zeros(block_shape+[1,1,1])
    center = np.asarray(mask.shape)//2
    it = np.nditer(mask, flags=['multi_index'])
    while not it.finished:
        x,y,z = it.multi_index
        it.iternext()
        cord = np.asarray([x,y,z])
        vector = (cord-center) * [16, 16, 40] + skel_vector*move_back
        distance = math.sqrt(sum(vector * vector)) + 1e-4
        if distance > radius + move_back:
            continue
        norm_vector = vector/distance
        if sum(norm_vector * skel_vector) < cos_theta and use_direct:
            continue
        mask[x,y,z] = 1
    candidates = np.setdiff1d(np.unique(mask*ffn_volume), 0)
    return np.asarray(candidates)


def stat():
    block_connector = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    csv_list = os.listdir(block_connector)
    stat_dict = {}
    for f in tqdm(csv_list):
        f_name = os.path.join(block_connector, f)
        df = pd.read_csv(f_name, index_col=0)
        stat_dict[f] = len(df)
    df = pd.DataFrame.from_dict(stat_dict, orient='index', columns=['num_connector'])
    df = df.reset_index().rename(columns={'index': 'block'})
    df = df.sort_values(by="num_connector", ascending=False)
    df.to_csv('/braindat/lab/liusl/flywire/block_data/v2/30_percent.csv', index=False)


def stat_neuron_connector():
    target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    connector_files = os.listdir(target_connector_path)
    valid_neuron_num = 0
    connector_num = []
    for connector_f in tqdm(connector_files):
        df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
        if len(df) > 10:
            valid_neuron_num = valid_neuron_num + 1
            connector_num.append(len(df))
    average_connector = sum(connector_num) / valid_neuron_num
    print(average_connector)


def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i * 2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i * 2) + 1].get_ydata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)


def plot():
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Unetft2/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Baseline_debugged/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/image_only_43k/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec4 = csv_list[1]
    acc4 = csv_list[2]
    labels = 'EdgeNetwork', 'Baseline', 'Ours-Embed.[14]', 'Ours-Image', 'Ours-Pairwise-Embed.'
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    ax1.set_title('Recall')
    bp = ax1.boxplot([rec0, rec2, rec3, rec4, rec1], showmeans=True, showfliers=False, labels=labels)
    print(get_box_plot_data(labels, bp))
    # ax1.set_yscale('log')
    ax1.set_ylim(0.77, 1.01)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    # ax1.set_xlim(0.62, 4.52)
    # show the ytick positions, as a reference

    ax2.set_title('Precision')
    bp = ax2.boxplot([acc0, acc2, acc3, acc4, acc1], showmeans=True, showfliers=False, labels=labels)
    print(get_box_plot_data(labels, bp))
    # ax2.set_yscale('log')
    ax2.set_ylim(0.77, 1.01)
    # ax2.set_xlim(0.63, 4.53)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    # ax1.text(1.5, 0.85, str(np.around(np.mean(rec0), decimals=3)), color="green", fontsize=10, ha='center')
    # ax1.text(2.27, 0.893, str(np.around(np.mean(rec2), decimals=3)), color="green", fontsize=10, ha='center')
    # ax1.text(3.27, 0.9, str(np.around(np.mean(rec3), decimals=3)), color="green", fontsize=10, ha='center')
    # ax1.text(4.27, 0.908, str(np.around(np.mean(rec1), decimals=3)), color="green", fontsize=10, ha='center')
    # ax2.text(1.27, 0.943, str(np.around(np.mean(acc0), decimals=3)), color="green", fontsize=10, ha='center')
    # ax2.text(2.27, 0.943, str(np.around(np.mean(acc2), decimals=3)), color="green", fontsize=10, ha='center')
    # ax2.text(3.27, 0.943, str(np.around(np.mean(acc3), decimals=3)), color="green", fontsize=10, ha='center')
    # ax2.text(4.27, 0.943, str(np.around(np.mean(acc1), decimals=3)), color="green", fontsize=10, ha='center')

    plt.savefig('box.pdf')


def set_new_threshold():
    # pred_path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/predictions'
    # csv_path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/thresh.csv'
    pred_path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/predictions'
    csv_path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/thresh.csv'
    preds = os.listdir(pred_path)
    threshes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for thresh in threshes:
        recall_list = []
        accuracy_list = []
        for pred in tqdm(preds):
            csv_list = pd.read_csv(os.path.join(pred_path, pred), header=None)
            gt = np.asarray(csv_list[2])
            prediction = np.asarray(csv_list[4])
            TP = sum(np.logical_and(prediction > thresh, gt == 1))
            FP = sum(np.logical_and(prediction > thresh, gt == 0))
            FN = sum(np.logical_and(prediction < thresh, gt == 1))
            recall = TP / (TP + FN)
            accuracy = TP / (TP + FP)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
        row = pd.DataFrame(
            [{'thresh': thresh, 'recall': np.mean(recall_list), 'accuracy': np.mean(accuracy_list)}])
        row.to_csv(csv_path, mode='a', header=False, index=False)


def plot_thresh():
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(6, 5.7))
    path = '/braindat/lab/liusl/flywire/experiment/test-flytracing/Baseline_debugged/0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-intensity/predictions0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-SegEmbed/predictions0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-Embed-2/predictions0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_pc0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec4 = csv_list[1]
    acc4 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Intensitytest_20480dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec5 = csv_list[1]
    acc5 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Metrictest0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec6 = csv_list[1]
    acc6 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Unettest0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec7 = csv_list[1]
    acc7 = csv_list[2]
    ax1.set_title('Misalignment', fontsize=12)
    line1, = ax1.plot(rec0, acc0, 'r-', label='EdgeNetwork')
    line2, = ax1.plot(rec1, acc1, 'b-', label='Edge+Intensity')
    line3, = ax1.plot(rec2, acc2, 'g-', label='Edge+Seg-Embed')
    line4, = ax1.plot(rec3, acc3, label='Edge+Connect-Embed', color='skyblue')
    line5, = ax1.plot(rec4, acc4, label='PointNet++', color='purple')
    line6, = ax1.plot(rec5, acc5, label='Point+Intensity', color='gray')
    line7, = ax1.plot(rec6, acc6, label='Point+Seg-Embed', color='pink')
    line8, = ax1.plot(rec7, acc7, label='Point+Connect-Embed', color='orange')
    ax1.plot(rec0[9], acc0[9], 'r^')
    ax1.plot(rec1[9], acc1[9], 'b^')
    ax1.plot(rec2[9], acc2[9], 'g^')
    ax1.plot(rec3[8], acc3[8], '^', color="skyblue")
    ax1.plot(rec4[9], acc4[9], '^', color="purple")
    ax1.plot(rec5[9], acc5[9], '^', color="gray")
    ax1.plot(rec6[9], acc6[9], '^', color="pink")
    ax1.plot(rec7[9], acc7[9], '^', color="orange")
    # ax1.legend(fontsize=11)
    ax1.set_ylim(0.84, 1.0)
    ax1.set_xlim(0.60, 1.0)
    # ax1.set_xlabel('recall', labelpad=0.4, fontsize=13)
    ax1.set_ylabel('precision', fontsize=12)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    path = '/braindat/lab/liusl/flywire/experiment/test-flytracing/Baseline_debugged/1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-intensity/predictions1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-SegEmbed/predictions1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-Embed-2/predictions1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_pc1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec4 = csv_list[1]
    acc4 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Intensitytest_20481dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec5 = csv_list[1]
    acc5 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Metrictest1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec6 = csv_list[1]
    acc6 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Unettest1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec7 = csv_list[1]
    acc7 = csv_list[2]
    ax2.set_title('Missing-Section', fontsize=12)
    ax2.plot(rec0, acc0, 'r-', label='EdgeNetwork')
    ax2.plot(rec1, acc1, 'b-', label='Edge+Intensity')
    ax2.plot(rec2, acc2, 'g-', label='Edge+Seg-Embed')
    ax2.plot(rec3, acc3, label='Edge+Connect-Embed', color='skyblue')
    ax2.plot(rec4, acc4, label='PointNet++', color='purple')
    ax2.plot(rec5, acc5, label='Point+Intensity', color='gray')
    ax2.plot(rec6, acc6, label='Point+Seg-Embed', color='pink')
    ax2.plot(rec7, acc7, label='Point+Connect-Embed', color='orange')
    ax2.plot(rec0[9], acc0[9], 'r^')
    ax2.plot(rec1[9], acc1[9], 'b^')
    ax2.plot(rec2[9], acc2[9], 'g^')
    ax2.plot(rec3[9], acc3[9], '^', color="skyblue")
    ax2.plot(rec4[9], acc4[9], '^', color="purple")
    ax2.plot(rec5[9], acc5[9], '^', color="gray")
    ax2.plot(rec6[9], acc6[9], '^', color="pink")
    ax2.plot(rec7[9], acc7[9], '^', color="orange")
    # ax2.legend()
    ax2.set_ylim(0.84, 1.0)
    ax2.set_xlim(0.60, 1.0)
    # ax2.set_xlabel('recall', labelpad=0.4, fontsize=13)
    # ax4.set_ylabel('precision', fontsize=12)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    path = '/braindat/lab/liusl/flywire/experiment/test-flytracing/Baseline_debugged/0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-intensity/predictions0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-SegEmbed/predictions0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-Embed-2/predictions0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_pc0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec4 = csv_list[1]
    acc4 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Intensitytest_20480.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec5 = csv_list[1]
    acc5 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Metrictest0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec6 = csv_list[1]
    acc6 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Unettest0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec7 = csv_list[1]
    acc7 = csv_list[2]
    ax3.set_title('Mixed', fontsize=12)
    ax3.plot(rec0, acc0, 'r-', label='EdgeNetwork')
    ax3.plot(rec1, acc1, 'b-', label='Edge+Intensity')
    ax3.plot(rec2, acc2, 'g-', label='Edge+Seg-Embed')
    ax3.plot(rec3, acc3, label='Edge+Connect-Embed', color='skyblue')
    ax3.plot(rec4, acc4, label='PointNet++', color='purple')
    ax3.plot(rec5, acc5, label='Point+Intensity', color='gray')
    ax3.plot(rec6, acc6, label='Point+Seg-Embed', color='pink')
    ax3.plot(rec7, acc7, label='Point+Connect-Embed', color='orange')
    ax3.plot(rec0[9], acc0[9], 'r^')
    ax3.plot(rec1[9], acc1[9], 'b^')
    ax3.plot(rec2[9], acc2[9], 'g^')
    ax3.plot(rec3[7], acc3[7], '^', color="skyblue")
    ax3.plot(rec4[9], acc4[9], '^', color="purple")
    ax3.plot(rec5[9], acc5[9], '^', color="gray")
    ax3.plot(rec6[9], acc6[9], '^', color="pink")
    ax3.plot(rec7[9], acc7[9], '^', color="orange")
    # ax2.legend()
    ax3.legend(handles=[line1, line2, line3, line4],
               labels=['EdgeNetwork', 'Edge+Intensity', 'Edge+Seg-Embed', 'Edge+Connect-Embed'], bbox_to_anchor=(0.56, -0.94), loc="lower center")
    ax3.set_ylim(0.84, 1.0)
    ax3.set_xlim(0.60, 1.0)
    ax3.set_xlabel('recall', labelpad=0.4, fontsize=12)
    ax3.set_ylabel('precision', fontsize=12)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    path = '/braindat/lab/liusl/flywire/experiment/test-flytracing/Baseline_debugged/dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-intensity/predictions2dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-SegEmbed/predictions2dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-Embed-2/predictions2dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_pc2dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec4 = csv_list[1]
    acc4 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Intensitytest_20482dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec5 = csv_list[1]
    acc5 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Metrictest2dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec6 = csv_list[1]
    acc6 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Unettest2dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec7 = csv_list[1]
    acc7 = csv_list[2]
    ax4.set_title('Average', fontsize=12)
    ax4.plot(rec0, acc0, 'r-', label='EdgeNetwork')
    ax4.plot(rec1, acc1, 'b-', label='Edge+Intensity')
    ax4.plot(rec2, acc2, 'g-', label='Edge+Seg-Embed')
    ax4.plot(rec3, acc3, label='Edge+Connect-Embed', color='skyblue')
    ax4.plot(rec4, acc4, label='PointNet++', color='purple')
    ax4.plot(rec5, acc5, label='Point+Intensity', color='gray')
    ax4.plot(rec6, acc6, label='Point+Seg-Embed', color='pink')
    ax4.plot(rec7, acc7, label='Point+Connect-Embed', color='orange')
    ax4.plot(rec0[9], acc0[9], 'r^')
    ax4.plot(rec1[9], acc1[9], 'b^')
    ax4.plot(rec2[9], acc2[9], 'g^')
    ax4.plot(rec3[8], acc3[8], '^', color="skyblue")
    ax4.plot(rec4[9], acc4[9], '^', color="purple")
    ax4.plot(rec5[9], acc5[9], '^', color="gray")
    ax4.plot(rec6[9], acc6[9], '^', color="pink")
    ax4.plot(rec7[9], acc7[9], '^', color="orange")
    ax4.legend(handles=[line5, line6, line7, line8],
               labels=['PointNet++', 'Point+Intensity', 'Point+Seg-Embed', 'Point+Connect-Embed'], bbox_to_anchor=(0.56, -0.94), loc="lower center")
    # ax2.legend()
    # ax4.legend(fontsize=11, bbox_to_anchor=(1.04, 0), loc="lower left")
    # ax4.legend(bbox_to_anchor=(0.57, -1.5), loc="lower center")
    # ax4.legend(loc="best")
    ax4.set_ylim(0.84, 1.0)
    ax4.set_xlim(0.60, 1.0)
    ax4.set_xlabel('recall', labelpad=0.4, fontsize=12)
    # ax4.set_ylabel('precision', fontsize=12)
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax4.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.18)
    plt.subplots_adjust(wspace=0.22, hspace=0.3)

    plt.savefig('/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/thresh.pdf')


def plot_thresh_distort():
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/image_distortion.xlsx'
    pred_path = ['/braindat/lab/liusl/flywire/experiment/test-3k/Edge-Embed-2/predictions',
                 '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-intensity/predictions',
                 '/braindat/lab/liusl/flywire/experiment/test-3k/Edge-SegEmbed/predictions',
                 '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Metrictest',
                 '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Unettest',
                 '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_pc',
                 '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/prediction_extract_pc_Intensitytest_2048']
    pred_path = ['/braindat/lab/liusl/flywire/experiment/test-3k/Edge-Embed-2/predictions']
    list = pd.read_excel(path, header=None)
    target_type = [2, 1, 0, 0.5]
    target_type = [0]
    threshes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # threshes = [0.5]
    for method in pred_path:
        for thresh in threshes:
            recall_list = []
            accuracy_list = []
            for block, t in zip(list[0], list[1]):
                if isinstance(block, str):
                    if t in target_type:
                        if os.path.exists(os.path.join(method, block)):
                            csv_list = pd.read_csv(os.path.join(method, block), header=None)
                            gt = np.asarray(csv_list[2])
                            prediction = np.asarray(csv_list[4])
                            TP = sum(np.logical_and(prediction > thresh, gt == 1))
                            FP = sum(np.logical_and(prediction > thresh, gt == 0))
                            FN = sum(np.logical_and(prediction < thresh, gt == 1))
                            recall = TP / (TP + FN)
                            accuracy = TP / (TP + FP)
                            recall_list.append(recall)
                            accuracy_list.append(accuracy)
            row = pd.DataFrame(
                [{'thresh': thresh, 'recall': np.mean(recall_list), 'accuracy': np.mean(accuracy_list)}])
            row.to_csv(method + str(target_type[0]) + 'dist_thresh.csv', mode='a', header=False,
                       index=False)


def stat_biological_recall():
    potential_path = '/braindat/lab/liusl/flywire/block_data/neuroglancer/connector_22_7_97-101.csv'
    block_paths = ['/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_97.csv',
                   '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_98.csv',
                   '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_99.csv',
                   '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_100.csv',
                   '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_101.csv']

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


def stat_positive_samples():
    filepath_data = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000'
    filenames = os.listdir(filepath_data)
    data_num = 0
    for f in filenames:
        path = os.path.join(filepath_data, str(f))
        df = pd.read_csv(path, header=None)
        data_num = data_num + len(df)
    print(data_num)


def delete_far():
    connector_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_vali/connector_14_7_167.csv'
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_vali_filtered/connector_14_7_167.csv'
    sample_volume_size = [16, 128, 128]
    connector_list = pd.read_csv(connector_path, header=None)
    idx = 0
    while idx < len(connector_list):
        connector = connector_list.iloc[idx]
        cord_start_offset = np.asarray(connector[3][1:-1].split(), dtype=np.float32)
        cord_pos_offset = np.asarray(connector[4][1:-1].split(), dtype=np.float32)
        cord_start_offset = np.asarray([cord_start_offset[0] / 4, cord_start_offset[1] / 4, cord_start_offset[2]])
        cord_pos_offset = np.asarray([cord_pos_offset[0] / 4, cord_pos_offset[1] / 4, cord_pos_offset[2]])
        if np.any(np.abs(cord_start_offset - cord_pos_offset) > np.asarray(
                [sample_volume_size[1] - 32, sample_volume_size[2] - 32, sample_volume_size[0] - 2],
                dtype=np.float32)):
            connector_list.drop(idx, inplace=True)
        idx = idx + 1
    connector_list.to_csv(target_path, header=False, index=False)


def plot_3d(volume):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    volume = resize(volume, [3, 64, 64, 64], order=0)
    voxels = volume[2, :, :, :] != 0
    colors = np.empty(voxels.shape, dtype=object)
    colors[volume[0, :, :, :] > 0] = 'red'
    colors[volume[1, :, :, :] > 0] = 'blue'
    ax.voxels(voxels, facecolors=colors)
    plt.show()


def merge_validation_files():
    validation_files_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/xray-validation-downsamplesamples'
    files = [os.path.join(validation_files_path, x) for x in os.listdir(validation_files_path)]
    merged_file = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/xray-validation-downsamplesamples.csv'
    for file in files:
        rows = pd.read_csv(file, header=None)
        rows.to_csv(merged_file, mode='a', header=False, index=False)


def fulse_augment():
    action = 'max'
    file1 = '/braindat/lab/liusl/flywire/experiment/test/x-ray-3000-axis0/prediction.csv'
    file2 = '/braindat/lab/liusl/flywire/experiment/test/x-ray-3000-axis1/prediction.csv'
    file3 = '/braindat/lab/liusl/flywire/experiment/test/x-ray-3000-axis2/prediction.csv'
    fulsed_file = '/braindat/lab/liusl/flywire/experiment/test/x-ray-3000_prediction-max-smooth.csv'
    edges1 = pd.read_csv(file1, header=None)
    edges2 = pd.read_csv(file2, header=None)
    edges3 = pd.read_csv(file3, header=None)
    for i in range(len(edges1)):
        if action == 'max':
            if np.std([edges1[4][i].item(), edges2[4][i].item(), edges3[4][i].item()]) > 0.35 and \
                    np.sort([edges1[4][i].item(), edges2[4][i].item(), edges3[4][i].item()])[0] < 0.1:
                value = np.mean([edges1[4][i].item(), edges2[4][i].item(), edges3[4][i].item()])
            else:
                value = np.max([edges1[4][i].item(), edges2[4][i].item(), edges3[4][i].item()])
            # if value > 0.999:
            #     value = np.mean([edges1[4][i].item(), edges2[4][i].item(), edges3[4][i].item()])
        elif action == 'mean':
            value = np.mean([edges1[4][i].item(), edges2[4][i].item(), edges3[4][i].item()])
        row = pd.DataFrame(
            [{'node0_segid': int(edges1[0][i]), 'node1_segid': int(edges1[1][i]),
              'target': int(-1),
              'prediction': int(-1), 'value': value}])
        row.to_csv(fulsed_file, mode='a', header=False, index=False)


def get_biological_samples():
    h5_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/edges-2000nm-128x128x128/testing/unknowns/xray-test-downsample-examples.h5'
    samples = readh5(h5_path)
    sample_list_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/xray-test-downsamplemapping_result.csv'
    edges = pd.read_csv(sample_list_path, header=None)
    candidate_path = '/braindat/lab/liusl/x-ray/test_candidates_2000nm'
    if os.path.exists(candidate_path):
        shutil.rmtree(candidate_path)
    os.mkdir(candidate_path)
    for i in range(len(edges)):
        sample_name = os.path.join(candidate_path, str(edges[0][i]) + '_' + str(edges[1][i]) + '.h5')
        if i == 1548:
            print('check')
        if not os.path.exists(sample_name):
            sample_array = samples[i, :, :, :]
            if len(np.unique(sample_array)) > 2:
                # three_channel_array = [np.zeros(sample_array.shape), np.zeros(sample_array.shape), np.zeros(sample_array.shape)]
                # three_channel_array[2][np.where(sample_array == 1)] = 1
                # three_channel_array[2][np.where(sample_array == 2)] = 1
                # three_channel_array[0][np.where(sample_array == 1)] = 1
                # three_channel_array[1][np.where(sample_array == 2)] = 1
                # three_channel_array = np.concatenate((np.expand_dims(three_channel_array[0], 0), np.expand_dims(three_channel_array[1], 0), np.expand_dims(three_channel_array[2], 0)))
                # writeh5(sample_name, three_channel_array)
                writeh5(sample_name, sample_array)


def plot_ablation_xray():
    thresh = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    value_w = [0.773, 0.780, 0.784, 0.787, 0.7908, 0.7915, 0.7855, 0.790, 0.7866]
    value_wo = [0.758, 0.758, 0.758, 0.758, 0.758, 0.758, 0.758, 0.743, 0.585]
    baseline = [0.758, 0.758, 0.758, 0.758, 0.758, 0.758, 0.758, 0.758, 0.758]
    fig = plt.figure(figsize=(6, 4.5))
    plt.plot(thresh, value_wo, label='without pretrain')
    plt.plot(thresh, value_w, 'r', label='with pretrain')
    plt.plot(thresh, baseline, '--', label='baseline')
    plt.ylabel('validation XPRESS score')
    plt.legend()
    plt.xlabel('threshold')
    plt.show()


def visualize_pc():
    import open3d as o3d
    path = r"‪F:\flywire数据\pc\1758404938_8122092051.ply"
    # 读取文件
    pcd = o3d.io.read_point_cloud(path)  # path为文件路径

    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, 1)
    o3d.visualization.draw_geometries([pcd_new])


def partition_once(skel, p_vectors, p_cords, method='branch'):
    """
    partition the skel once and record partition information
    Input: segment skel to partition, partition cords & vectors
    parameters: method = 'branch' or 'longest_path'
    Returns: partitioned skel

    """
    if method == 'branch':
        main_branchpoint = navis.find_main_branchpoint(skel, method='longest_neurite')
        main_branchpoint = main_branchpoint[0] if isinstance(main_branchpoint, list) else main_branchpoint
        childs = list(skel.nodes[skel.nodes.parent_id == main_branchpoint].node_id.values)
        assert len(childs) == 2, "branch point contains more than 2 child"
        try:
            split = navis.cut_skeleton(skel, [childs[0], main_branchpoint])
        except:
            skel = navis.reroot_skeleton(skel, childs[1])
            split = navis.cut_skeleton(skel, [childs[0], main_branchpoint])
        child = childs[0]
        grand_child = skel.nodes[skel.nodes.parent_id == child].node_id.values[0]
        p_cords_child = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['y'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['z'][skel.nodes['node_id'] == child].item() / 40])
        p_cords_grandchild = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['y'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['z'][skel.nodes['node_id'] == grand_child].item() / 40])
        p_vector = p_cords_grandchild - p_cords_child
        p_cord = (p_cords_child + p_cords_grandchild) / 2
        p_vectors.append(p_vector)
        p_cords.append(p_cord)

        child = childs[1]
        grand_child = skel.nodes[skel.nodes.parent_id == child].node_id.values[0]
        p_cords_child = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['y'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['z'][skel.nodes['node_id'] == child].item() / 40])
        p_cords_grandchild = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['y'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['z'][skel.nodes['node_id'] == grand_child].item() / 40])
        p_vector = p_cords_grandchild - p_cords_child
        p_cord = (p_cords_child + p_cords_grandchild) / 2
        p_vectors.append(p_vector)
        p_cords.append(p_cord)

        cut_point = main_branchpoint

    elif method == 'longest_path':
        # Find the two most distal points
        leafs = skel.leafs.node_id.values
        dists = navis.geodesic_matrix(skel, from_=leafs)[leafs]

        # This might be multiple values
        mx = np.where(dists == np.max(dists.values))
        start = dists.columns[mx[0][0]]

        # Reroot to one of the nodes that gives the longest distance
        skel.reroot(start, inplace=True)
        if isinstance(skel, navis.NeuronList): skel = skel[0]
        segments = navis.graph_utils._generate_segments(skel, weight='weight')
        longest_path = segments[0]
        cut_point = longest_path[int(len(longest_path)/2)]
        split = navis.cut_skeleton(skel, cut_point)

        child = cut_point
        grand_child = skel.nodes[skel.nodes.parent_id == child].node_id.values[0]
        p_cords_child = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['y'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['z'][skel.nodes['node_id'] == child].item() / 40])
        p_cords_grandchild = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['y'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['z'][skel.nodes['node_id'] == grand_child].item() / 40])
        p_vector = p_cords_grandchild - p_cords_child
        p_cord = (p_cords_child + p_cords_grandchild) / 2
        p_vectors.append(p_vector)
        p_cords.append(p_cord)

    else:
        raise ValueError("Invalid method")

    # visulize cut point
    fig, ax = navis.plot2d(skel, method='2d', view=('x', '-y'))
    cut_coords = skel.nodes.set_index('node_id').loc[cut_point, ['x', 'y']].values
    ax.annotate('cut point',
                xy=(cut_coords[0], -cut_coords[1]),
                color='red',
                xytext=(cut_coords[0], -cut_coords[1] - 2000), va='center', ha='center',
                arrowprops=dict(shrink=0.1, width=2, color='red'),
                )
    plt.show()

    return split


def segment_partition():
    """
    Input: segment id, partition cords & vectors
    Returns: partition segment point cloud

    """
    navis.patch_cloudvolume()
    segment_id = 4098055229
    vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True)  #
    vol_ffn1.parallel = 8
    vol_ffn1.meta.info['skeletons'] = 'skeletons_32nm'
    vol_ffn1.skeleton.meta.refresh_info()
    vol_ffn1.skeleton.meta.info['sharding']['hash'] = 'murmurhash3_x86_128'
    vol_ffn1.skeleton = ShardedPrecomputedSkeletonSource(vol_ffn1.skeleton.meta, vol_ffn1.cache, vol_ffn1.config)
    skel = vol_ffn1.skeleton.get(segment_id, as_navis=True)

    # cut the first and second child with their grand child
    # raw voxel cord
    p_vectors = []
    p_cords = []
    skel_queue = Queue(maxsize=0)
    skel_queue.put(skel)
    while not skel_queue.empty():
        skel = skel_queue.get()
        splits = partition_once(skel, p_vectors, p_cords, method='branch')
        for s in splits:
            s.to_swc(os.path.join('/braindat/lab/liusl/flywire/log', str(s.cable_length) + '.swc'))
        # assert len(splits) == 3, "result in more than three branches"
        for split in splits:
            if split.cable_length > 100000:
                skel_queue.put(split)

    print(p_cords, p_vectors)


from scipy.stats import gaussian_kde

def get_density_based_bounding_box(points, percentage):
    # Convert the list of points to a numpy array
    points_array = np.array(points)

    # Calculate the number of points to be included (30% of the total points)
    num_points_to_include = int(len(points) * percentage)

    # Calculate the density of the points using kernel density estimation
    kde = gaussian_kde(points_array.T)
    density = kde(points_array.T)

    # Sort the points based on density (higher density first)
    sorted_indices = np.argsort(density)[::-1]
    points_sorted_by_density = points_array[sorted_indices]
    # pickle.dump(points_sorted_by_density,  open('/braindat/lab/liusl/flywire/fafb-public-skeletons/point_density.pickle', 'wb'))

    # Get the points corresponding to the top density values (highest density region)
    points_subset = points_sorted_by_density[:num_points_to_include]

    # Find the minimum and maximum coordinates along each axis for the subset of points
    min_coords = np.min(points_subset, axis=0)
    max_coords = np.max(points_subset, axis=0)

    # Compute the dimensions of the bounding box
    bounding_box_dimensions = max_coords - min_coords

    return bounding_box_dimensions, min_coords, max_coords, points_subset

def get_proofread_blocks():
    """
    Input: neurons for evaluation, networkx format
    Returns: 20 blocks with the most nodes, to be proofread

    """
    graph = pickle.load(open('/braindat/lab/liusl/flywire/fafb-public-skeletons/skeleton_segment_graph.pickle', 'rb'))
    points = []
    block_dict = {}
    for node, attr in tqdm(graph.nodes(data=True)):
        # point = attr['zyx_coord']/np.array([4,4,40])
        point = attr['zyx_coord']
        points.append(point)
    random.shuffle(points)
    box, box_min, box_max, points_subset = get_density_based_bounding_box(points[:100000], 0.1)
    for point in points_subset:
        coord = point / np.array([4, 4, 40])
        x_block, y_block, z_block = fafb_to_block(coord[0], coord[1], coord[2])
        if str(x_block) + '_' + str(y_block) + '_' + str(z_block) in block_dict.keys():
            block_dict[str(x_block) + '_' + str(y_block) + '_' + str(z_block)] += 1
        else:
            block_dict[str(x_block) + '_' + str(y_block) + '_' + str(z_block)] = 1
    print(len(block_dict))
    pickle.dump(block_dict,
                open('/braindat/lab/liusl/flywire/fafb-public-skeletons/top_30_blocks.pickle', 'wb'))
    #     x_block, y_block, z_block = fafb_to_block(point[0], point[1], point[2])
    #     if block_dict[str(x_block)+'_'+str(y_block)+'_'+str(z_block)] is not None:
    #         block_dict[[x_block, y_block, z_block]] += 1
    #     else:
    #         block_dict[[x_block, y_block, z_block]] = 1
    # sorted(block_dict)

def get_evaluate_blocks():
    filepath_data = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/'
    block_csv_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/block'
    indexes = range(0,16)
    edge_number = 0
    for index in indexes:
        file_name = 'evaluate_fafb_dust1200_dis500_rad0.3216_' + str(index) + '.csv'
        df = pd.read_csv(os.path.join(filepath_data, file_name), header=None)
        N = len(df)
        edge_number += N
        for ii in tqdm(range(0, N)):
            segid1 = df.iloc[ii, 0]
            segid2 = df.iloc[ii, 1]
            cord = df.iloc[ii, 2][1:-1].split(' ')
            cord = [item for item in cord if item != ""]
            x,y,z= [float(cord[0]), float(cord[1]), float(cord[2])]
            x_block, y_block, z_block = fafb_to_block(x, y, z)
            block_name = 'connector_' + str(x_block) + '_' + str(y_block) + '_' + str(z_block)
            row = pd.DataFrame(
                [{'node0_segid': int(segid1), 'node1_segid': int(segid2), 'cord': cord, 'target': -1,
                  'prediction': -1, 'box_index': index}])
            row.to_csv(os.path.join(block_csv_path, block_name+ '.csv'), mode='a', header=False, index=False)
    print('total edge number %d'%edge_number)


def get_zebrafinch():
    # annotation_path = '/braindat/zebrafinch/GT/gt_z255-383_y1407-1663_x1535-1791.h5'
    # seg = readh5(annotation_path)
    file_names = [
        "gt_z255-383_y1407-1663_x1535-1791.h5",
        "gt_z2559-2687_y4991-5247_x4863-5119.h5",
        "gt_z2815-2943_y5631-5887_x4607-4863.h5",
        "gt_z2834-2984_y5311-5461_x5077-5227.h5",
        "gt_z2868-3018_y5744-5894_x5157-5307.h5",
        "gt_z2874-3024_y5707-5857_x5304-5454.h5",
        "gt_z2934-3084_y5115-5265_x5140-5290.h5",
        "gt_z3096-3246_y5954-6104_x5813-5963.h5",
        "gt_z3118-3268_y6538-6688_x6100-6250.h5",
        "gt_z3126-3276_y6857-7007_x5694-5844.h5",
        "gt_z3436-3586_y599-749_x2779-2929.h5",
        "gt_z3438-3588_y2775-2925_x3476-3626.h5",
        "gt_z3456-3606_y3188-3338_x4043-4193.h5",
        "gt_z3492-3642_y7888-8038_x8374-8524.h5",
        "gt_z3492-3642_y841-991_x381-531.h5",
        "gt_z3596-3746_y3888-4038_x3661-3811.h5",
        "gt_z3604-3754_y4101-4251_x3493-3643.h5",
        "gt_z3608-3758_y3829-3979_x3423-3573.h5",
        "gt_z3702-3852_y9605-9755_x2244-2394.h5",
        "gt_z3710-3860_y8691-8841_x2889-3039.h5",
        "gt_z3722-3872_y4548-4698_x2879-3029.h5",
        "gt_z3734-3884_y4315-4465_x2209-2359.h5",
        "gt_z3914-4064_y9035-9185_x2573-2723.h5",
        "gt_z4102-4252_y6330-6480_x1899-2049.h5",
        "gt_z4312-4462_y9341-9491_x2419-2569.h5",
        "gt_z4440-4590_y7294-7444_x2350-2500.h5",
        "gt_z4801-4951_y10154-10304_x1972-2122.h5",
        "gt_z4905-5055_y928-1078_x1729-1879.h5",
        "gt_z4951-5101_y9415-9565_x2272-2422.h5",
        "gt_z5001-5151_y9426-9576_x2197-2347.h5",
        "gt_z5119-5247_y1023-1279_x1663-1919.h5",
        "gt_z5405-5555_y10490-10640_x3406-3556.h5",
        "gt_z734-884_y9561-9711_x563-713.h5"
    ]

    for file_name in file_names:
        # 使用字符串分割操作提取 Z、Y、X 范围
        match = re.search(r"z(\d+)-(\d+)_y(\d+)-(\d+)_x(\d+)-(\d+)", file_name)

        if match:
            # 提取范围值
            z_start, z_end, y_start, y_end, x_start, x_end = map(int, match.groups())
        else:
            print(file_name)
        # 打印范围
        print(f"ZSTART, ZEND = {z_start}, {z_end}")
        print(f"YSTART, YEND = {y_start}, {y_end}")
        print(f"XSTART, XEND = {x_start}, {x_end}")
        print()
        vol = CloudVolume(
            'https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata',
            mip=0, bounded=True, progress=False)
        image_xyzc = vol[x_start:x_end, y_start:y_end, z_start:z_end]
        image_zyx = np.transpose(image_xyzc[..., 0], [2, 1, 0])
        name = 'image_' + 'z' + str(z_start) + '-' + str(z_end) + '_'+ 'y' + str(y_start) + '-' + str(y_end) + 'x' + str(x_start) + '-' + str(x_end) + '.h5'
        writeh5('/braindat/zebrafinch/GT/'+name, image_zyx)


def get_connection_graph():
    result_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/block_result'
    thresh = 0.95
    train_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train_1000'
    # 创建一个无向图
    G = nx.Graph()
    edge_num = 0
    csv_names = os.listdir(result_path)
    train_name = os.listdir(train_path)
    for csv_name in tqdm(csv_names):
        if csv_name not in train_name:
            try:
                df = pd.read_csv(os.path.join(result_path, csv_name), header=None)
                edge_num += len(df)
                threshed_df = df[df.iloc[:, 4] > thresh]
                for ii, row in threshed_df.iterrows():
                    # 添加边到图中
                    try:
                        G.add_edge(int(row[0]), int(row[1]))
                    except:
                        print('debug')
            except:
                print(csv_name)
                os.remove(os.path.join(result_path, csv_name))
        else:
            print('%s is in train'%csv_name)
    print('total edge number %d'%edge_num)
    subgraphs = list(nx.connected_components(G))
    pickle.dump(subgraphs, open('/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/result_%.3f.pickle'%thresh, 'wb'))


def get_prediction_distribution():
    path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/block_result'
    plot_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/plot'
    csv_names = os.listdir(path)
    for csv_name in tqdm(csv_names):
        df = pd.read_csv(os.path.join(path, csv_name), header=None)
        prediction_list = []
        for i in range(len(df)):
            prediction_list.append(df.iloc[i,4])
        prediction_list = np.asarray(prediction_list)
        name = csv_name.split('.')[0].split('_')[1:]
        name = name[0] + '_' + name[1] + '_' +name[2]
        plt.figure(figsize=(4, 3))
        plt.hist(prediction_list, bins=10, alpha=0.7, color='blue', edgecolor='black')
        plt.gca().set_yscale('log')
        # plt.ylim([1e3, 3e4])  # 设置y轴的上下限
        plt.xlabel('Prediction')
        plt.ylabel('Count (log scale)')
        plt.title('block %s'%name)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(plot_path, name+'.pdf'))

def connection_score_plot():
    target_tree_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data'
    target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    visualization_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/visualization'
    connector_files = os.listdir(target_connector_path)
    total_length = 0
    edge_num = 0
    score_num_dict = {}
    score_length_dict = {}
    for connector_f in tqdm(connector_files[:200]):
        # block_list = define_block_list()
        visualization_f = connector_f.replace('_connector.csv', '_save.csv')
        neuron_id = visualization_f[:-9]
        print('processing neuron', neuron_id)
        df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
        df_weight = pd.read_csv(os.path.join(visualization_path, visualization_f), index_col=0)
        if len(df) > 10:
            tree_f = connector_f.replace('_connector.csv', '.json')
            tree = navis.read_json(os.path.join(target_tree_path, tree_f))[0]
            max_strahler_index = tree.nodes.strahler_index.max()

            for i in df.index:
                weight1 = df_weight[str(int(df['node1_segid'][i]))][0]
                weight0 = df_weight[str(int(df['node0_segid'][i]))][0]
                score1 = 1 / ((1 / min(weight1, 40)) * (1 / min(weight0, 40)))
                score2 = (1 / (max_strahler_index - int(df['Strahler order'][i].split('\n')[0].split(' ')[-1]) + 3))
                score = score2 * score1
                total_length += weight0 + weight1
                edge_num +=1
                if not score in score_num_dict.keys():
                    score_num_dict[score] = 1
                    score_length_dict[score] = (weight0+weight1)
                else:
                    score_num_dict[score] += 1
                    score_length_dict[score] += 2*(weight0 * weight1)/(weight0 + weight1)
        else:
            print('invalid neuron', neuron_id)
    x_values = list(score_num_dict.keys())
    y_values = [score_length_dict[x] for x in x_values]
    scatter_sizes = [score_num_dict[x] for x in x_values]

    # # 创建散点图
    # plt.figure(figsize=(8, 6))  # 图片大小为(8, 6)
    # plt.scatter(x_values, y_values, s=scatter_sizes, alpha=0.7, edgecolors='w')
    #
    # # 添加标题和标签
    # plt.title('Scatter Plot', fontsize=16)
    # plt.xlabel('Connection score', fontsize=14)
    # plt.ylabel('Skeleton coverage ratial', fontsize=14)
    # plt.gca().set_xscale('log')
    #
    # # 显示图例
    # plt.legend(['Data Points'], loc='upper left')
    #
    # # 显示网格线
    # plt.grid(True)

    fig, ax1 = plt.subplots(figsize=(6, 3))

    # 绘制第一个直方图
    custom_bins = [0.1, 0.18, 0.33, 0.59, 1, 1.8, 3.3, 5.94, 10, 18, 33, 59.4, 100, 180, 330, 590, 1000]
    color = 'tab:blue'
    hist, bins, _ = ax1.hist(x_values, bins=custom_bins, weights=scatter_sizes, alpha=0.7, color=color, edgecolor='black')
    ax1.set_xlabel('Connection score')
    ax1.set_ylabel('Count', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 在同一子图上绘制第二个直方图
    ax2 = ax1.twinx()  # 共享x轴
    color = 'tab:orange'
    hist, bins, _ = ax2.hist(x_values, bins=custom_bins, weights=y_values, alpha=0.7, color=color,
                             edgecolor='black')
    ax2.set_ylabel('Sum of increase score in the bin', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.gca().set_xscale('log')

    fig.tight_layout()
    # 显示图像
    plt.show()


def get_egde_network_f1():
    file_name = f'/braindat/lab/liuyixiong/data/biologicalgraphs/tmp_output/SNEMI3D_test_pairs_info_1103.csv'
    df = pd.read_csv(file_name)
    N = len(df)
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for ii in tqdm(range(0, N)):
        label = df.iloc[ii, 5]
        pred = df.iloc[ii, 4]
        if label == 0 and pred == 0:
            TN += 1
        if label == 1 and pred == 0:
            FN += 1
        if label == 0 and pred == 1:
            FP += 1
        if label == 1 and pred == 1:
            TP += 1
    recall = TP/(TP+FN)
    acc = TP/(TP+FP)
    f1 = 2*recall*acc/(recall+acc)
    print(recall,acc,f1)