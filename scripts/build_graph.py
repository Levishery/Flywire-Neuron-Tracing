import math
import time
import random
import struct
import skeletor as sk
import os
from tqdm import tqdm
import pandas as pd
from cloudvolume import Skeleton
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
from cloudvolume import CloudVolume
import cloudvolume
from PIL import Image

import numpy as np



def compute_normalized_vector(p_1, p_2):
    end_point_vector = p_1 - p_2
    norm = np.sqrt(sum(end_point_vector * end_point_vector))
    return end_point_vector/norm

def xpblock_to_fafb(z_block, y_block, x_block, z_coo, y_coo, x_coo):
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

def fafb_to_block(x, y, z):
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
    return x_block, y_block, z_block, x_pixel,y_pixel,z_pixel

def bio_constrained_candidates(skel_vector, end_point, seg_start, radius=500, theta=30, move_back=100):
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
        if sum(norm_vector * skel_vector) < cos_theta:
            continue
        mask[x,y,z] = 1
    candidates = np.setdiff1d(np.unique(mask*ffn_volume), 0)
    return candidates

def download_fafb(estimated_connection_points):
    fafb_v14 = cloudvolume.CloudVolume('https://storage.googleapis.com/neuroglancer-fafb-data/fafb_v14/fafb_v14_orig/',
                                       mip=2, progress=True)
    for estimated_connection_point in estimated_connection_points:
        cord = estimated_connection_point
        block_x, block_y, block_z, x_pixel, y_pixel, z_pixel = fafb_to_block(float(cord[0]), float(cord[1]),
                                                                             float(cord[2]))
        dir_name = 'connector_' + str(block_x) + '_' + str(block_y) + '_' + str(block_z)
        if not os.path.exists(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14', dir_name)):
            block_x = int(block_x)
            block_y = int(block_y)
            block_z = int(block_z)
            (start_x, start_y, start_z) = xpblock_to_fafb(block_z, block_y, block_x, 29, 0, 0)
            (end_x, end_y, end_z) = xpblock_to_fafb(block_z, block_y, block_x, 54, 1735, 1735)
            (start_x, start_y, start_z) = (start_x / 4 - 156, start_y / 4 - 156, start_z - 29)
            (end_x, end_y, end_z) = (end_x/4+156+1, end_y/4+156+1, end_z+29+1)
            print('begin dowload: ', dir_name)
            try:
                volume = fafb_v14[start_x:end_x, start_y:end_y, start_z:end_z]
            except:
                print('fail to download', dir_name)
                continue
            os.makedirs(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14', dir_name))
            for i in range(84):
                im = Image.fromarray(volume[:, :, i, 0])
                im.save(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14', os.path.join(dir_name, str(i).zfill(4)+'.png')))


vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True)  #
vol_ffn1.parallel = 8
vol_ffn1.meta.info['skeletons'] = 'skeletons_32nm'
vol_ffn1.skeleton.meta.refresh_info()
vol_ffn1.skeleton.meta.info['sharding']['hash'] = 'murmurhash3_x86_128'
vol_ffn1.skeleton = ShardedPrecomputedSkeletonSource(vol_ffn1.skeleton.meta, vol_ffn1.cache, vol_ffn1.config)
radius = 500

neuron_id = 720575940630120901
seg_start_path = '/braindat/lab/liusl/flywire/test-skel/segment_start'
block_path = os.path.join('/braindat/lab/liusl/flywire/test-skel/block_data/', str(neuron_id))
f_name = os.path.join(seg_start_path, str(neuron_id)+'.csv')
df = pd.read_csv(f_name, header=None)
seg_start_ids = np.unique(np.append(np.asarray(df[0]), np.asarray(df[1])))
skels, missing = vol_ffn1.skeleton.get(seg_start_ids)
meshes = vol_ffn1.mesh.get(missing)

end_point_vectors = []
estimated_connection_points = []
seg_start_list = []
seg_candidate_list = []
for seg_start in tqdm(missing+skels):
    radius = 400
    # radius = 20
    if isinstance(seg_start, np.int64):
        mesh = meshes[seg_start]
        try:
            # fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=True)
            skel = sk.skeletonize.by_wavefront(mesh, waves=1, step_size=2, progress=False)
            skel.save_swc('save.swc')
            with open('save.swc', 'r') as f_swc:
                x = f_swc.read()
            skel = Skeleton.from_swc(x)
            skel.id = seg_start
            seg_start = skel
        except:
            print('cannot skeletonize ', id)
            continue
    paths = seg_start.paths()
    average_connection_distance = seg_start.cable_length()/len(seg_start.edges)
    end_points_x = []
    if seg_start.id == 5520801401:
        print('.')
    for path in paths:
        terminal_points = path[:4]
        end_point_vector = compute_normalized_vector(terminal_points[0], terminal_points[-1])
        end_point = terminal_points[0]
        if end_point[0] not in end_points_x:
            end_points_x.append(end_point[0])
            end_point_vectors.append(end_point_vector)
            estimated_connection_point = end_point + end_point_vector*average_connection_distance
            estimated_connection_points.append(estimated_connection_point/[4,4,40])
            seg_start_list.append(seg_start.id)
            seg_candidate_list.append(bio_constrained_candidates(end_point_vector, end_point, seg_start.id, radius=radius))

        terminal_points = path[-4:]
        end_point_vector = compute_normalized_vector(terminal_points[-1], terminal_points[0])
        end_point = terminal_points[-1]
        if end_point[0] not in end_points_x:
            end_points_x.append(end_point[0])
            end_point_vectors.append(end_point_vector)
            estimated_connection_point = end_point + end_point_vector*average_connection_distance
            estimated_connection_points.append(estimated_connection_point/[4,4,40])
            seg_start_list.append(seg_start.id)
            seg_candidate_list.append(bio_constrained_candidates(end_point_vector, end_point, seg_start.id, radius=radius))

# df = pd.DataFrame(columns=['seg_start', 'estimated_connection_point', 'seg_candidate', 'block'])
for i in range(len(seg_start_list)):
    cord = estimated_connection_points[i]
    block_x, block_y, block_z, x_pixel, y_pixel, z_pixel = fafb_to_block(float(cord[0]), float(cord[1]),
                                                                         float(cord[2]))
    block = 'connector_' + str(block_x) + '_' + str(block_y) + '_' + str(block_z) + '.csv'
    row = pd.DataFrame(
        [{'node0_segid': int(seg_start_list[i]), 'node1_segid': np.array2string(seg_candidate_list[i].astype(np.int64)),
          'cord': np.array2string(cord), 'cord0': np.array2string(cord), 'cord1': np.array2string(cord), 'score': 0, 'neuron_id': neuron_id}])
    row.to_csv(os.path.join(block_path, block), mode='a', header=False, index=False)
    # df.loc[i] = [seg_start_list[i], estimated_connection_points[i], seg_candidate_list[i], block]
# df = df.sort_values(by='block')
# download fafb data if necessary
# download_fafb(estimated_connection_points)
# # load the model
# model = []
# for i in range(len(df)):





