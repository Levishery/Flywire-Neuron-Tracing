import matplotlib.pyplot as plt
import fafbseg
import navis
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from cilog import create_logger
from cloudvolume import CloudVolume
from flywire2fafbffn1_main import prune_cell_body_fiber
import os
import argparse
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Get GT voxel blocks for one neuron.")
    parser.add_argument('--neuron_id', type=str, default='720575940630120901',
                        help='neuron_id to be download')
    args = parser.parse_args()
    return args


def get_neuron_blocks():
    fafbseg.flywire.set_chunkedgraph_secret("5814c407f6cca6b2c1e50f5f99e6a555")
    create_logger(name='l1', file='/flywire_data//flywire2fafbffn1.log', sub_print=True)
    vol = CloudVolume('graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31', use_https=True, mip=0,
                      secrets='5814c407f6cca6b2c1e50f5f99e6a555')
    args = get_args()
    neuron_id = int(args.neuron_id)
    s = fafbseg.flywire.skeletonize_neuron(neuron_id, threads=1)
    s = prune_cell_body_fiber(s)
    s.nodes = navis.xform_brain(s.nodes, source='FLYWIRE', target='FAFB14raw', verbose=False)

    # statistic blocks in space fafbv14
    block_node_dict = {}
    for node in range(s.n_nodes):
        x_b, y_b, z_b = fafb_to_block(s.nodes['x'][node], s.nodes['y'][node], s.nodes['z'][node])
        block_id = 'connector_' + str(x_b) + '_' + str(y_b) + '_' + str(z_b)
        if block_id in block_node_dict:
            block_node_dict[block_id].append(node)
        else:
            block_node_dict[block_id] = [node]

    # get gt in space flywire by bbox
    for block in block_node_dict:
        if len(block_node_dict[block]) > 10:
            points = s.nodes.iloc[block_node_dict[block]]
            flywire_points = navis.xform_brain(points, source='FAFB14raw', target='FLYWIREraw', verbose=False)
            bbox = np.min(flywire_points['x']), np.max(flywire_points['x']), np.min(flywire_points['y']), \
                np.max(flywire_points['y']), np.min(flywire_points['z']), np.max(flywire_points['z'])
            flywire_block = vol[bbox[0] / 4 - 6: bbox[1] / 4 + 7, bbox[2] / 4 - 6: bbox[3] / 4 + 7,
                            bbox[4] - 3: bbox[5] + 4]
            supervoxel_ids = np.unique(flywire_block)
            root_ids = fafbseg.flywire.supervoxels_to_roots(supervoxel_ids)
            neuron_supervoxels = np.asarray(supervoxel_ids)[np.where(root_ids == neuron_id)]
            for super_voxel in neuron_supervoxels:
                neuron_voxels = np.where(flywire_block == super_voxel)
            neuron_voxels_fafbv14 = navis.xform_brain(neuron_voxels)


def get_one_block():
    fafbseg.flywire.set_chunkedgraph_secret("5814c407f6cca6b2c1e50f5f99e6a555")
    create_logger(name='l1', file='/braindat/lab/liusl/flywire/block_data/get_test_flywire_gt.log', sub_print=True)
    vol = CloudVolume('graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31', use_https=True, mip=0,
                      secrets='5814c407f6cca6b2c1e50f5f99e6a555')
    block_name = 'connector_23_7_58'
    [block_x, block_y, block_z] = block_name.split('_')[1:]
    [block_x, block_y, block_z] = [int(block_x), int(block_y), int(block_z)]
    points = [xpblock_to_fafb(block_z, block_y, block_x, 29, 0, 0),
              xpblock_to_fafb(block_z, block_y, block_x, 29, 0, 1735),
              xpblock_to_fafb(block_z, block_y, block_x, 29, 1735, 0),
              xpblock_to_fafb(block_z, block_y, block_x, 29, 1735, 1735),
              xpblock_to_fafb(block_z, block_y, block_x, 54, 0, 0),
              xpblock_to_fafb(block_z, block_y, block_x, 54, 0, 1735),
              xpblock_to_fafb(block_z, block_y, block_x, 54, 1735, 0),
              xpblock_to_fafb(block_z, block_y, block_x, 54, 1735, 1735)]
    flywire_points = navis.xform_brain(points, source='FAFB14raw', target='FLYWIREraw', verbose=False)
    bbox = np.min(flywire_points[:, 0]), np.max(flywire_points[:, 0]), np.min(flywire_points[:, 1]), \
        np.max(flywire_points[:, 1]), np.min(flywire_points[:, 2]), np.max(flywire_points[:, 2])
    flywire_block = vol[bbox[0] / 4: bbox[1] / 4 + 1, bbox[2] / 4: bbox[3] / 4 + 1, bbox[4]: bbox[5] + 1]
    supervoxel_ids = np.unique(flywire_block)
    root_ids = fafbseg.flywire.supervoxels_to_roots(supervoxel_ids)
    proofread = fafbseg.flywire.is_proofread(root_ids)
    block = np.zeros([1738, 1738, 28], dtype=np.int64)
    for i in tqdm(range(len(supervoxel_ids))):
        if proofread[i]:
            neuron_voxels_block = np.asarray(np.where(flywire_block == supervoxel_ids[i]))[:-1, :].transpose(1, 0)
            neuron_voxels_flywire = neuron_voxels_block * [4, 4, 1] + [bbox[0], bbox[2], bbox[4]]
            neuron_voxels_fafbv14 = navis.xform_brain(neuron_voxels_flywire, target='FAFB14raw', source='FLYWIREraw')
            # to prevent out-of-bound, clip the coordinate to a one voxel boundary
            neuron_voxels_block = (neuron_voxels_fafbv14 - points[0]) * [0.25, 0.25, 1] + [1, 1, 1]
            neuron_voxels_block = np.concatenate([np.clip(np.expand_dims(neuron_voxels_block[:, 0], axis=1), 0, 1737),
                                                  np.clip(np.expand_dims(neuron_voxels_block[:, 1], axis=1), 0, 1737),
                                                  np.clip(np.expand_dims(neuron_voxels_block[:, 2], axis=1), 0, 27)],
                                                 axis=1).astype(int)
            block[neuron_voxels_block] = root_ids[i]
    block = block[1:-1, 1:-1, 1:-1]
    with open('/braindat/lab/liusl/flywire/block_data/test_flywire_gt/%s.npy' % block_name, 'wb') as f:
        np.save(f, block)


get_one_block()
