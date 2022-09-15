import fafbseg
import navis
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import networkx as nx
from cloudvolume import CloudVolume
from sklearn.neighbors import NearestNeighbors
from cilog import create_logger
import time
from utils import default_dump
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
import pickle
from matplotlib import pyplot as plt
import os
import json


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


def prune_cell_body_fiber(x, method='betweenness', reroot_soma: bool = True, heal: bool = True,
                          threshold: float = 0.95, inplace: bool = False):
    """Prune neuron cell body fiber.

    Here, "cell body fiber" (CBF) refers to the tract connecting the soma to the
    backbone in unipolar neuron (common in e.g. insects). This function works
    best for typical neurons with clean skeletons.

    """

    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, navis.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron(s), got {type(x)}')

    if not inplace:
        x = x.copy()

    if x.n_trees > 1 and heal:
        _ = navis.heal_skeleton(x, method='LEAFS', inplace=True)

    # If no branches, just return the neuron
    if 'branch' not in x.nodes.type.values:
        return x

    if reroot_soma and not isinstance(x.soma, type(None)):
        x.reroot(x.soma, inplace=True)

    # Find main branch point
    cut = navis.find_main_branchpoint(x, method=method, threshold=threshold,
                                      reroot_soma=False)

    # Find the path to root (and account for multiple roots)

    for r in x.root:
        try:
            path = nx.shortest_path(x.graph, target=r, source=cut)
            break
        except nx.NetworkXNoPath:
            continue
        except BaseException:
            raise
    cut = path[1]
    subts = navis.cut_skeleton(x, cut)

    return subts[0]


def prune_to_backbone(x):
    x = prune_cell_body_fiber(x)
    _ = navis.strahler_index(x)
    x = x.prune_by_strahler(to_prune=range(1, x.nodes.strahler_index.max()))
    return x


def get_mapped_flywire_neuron(neuron_id, prune='cell_body_fiber'):
    """
    :param neuron_id:
    :return: mapped tree neurons of neuron_id, from flywire to fafbv14
    """
    # swc_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/swc'
    # mapped_point_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/mapped_points'
    skels = fafbseg.flywire.skeletonize_neuron_parallel(neuron_id, n_cores=4)
    # s.to_swc(os.path.join(swc_path, str(neuron_id) + '.swc'))
    mapped_skels = []
    for s in skels:
        if s.soma is None:
            print(f'#W#neuron {s.id} has no soma record, skip.')
            continue
        try:
            s.nodes = navis.xform_brain(s.nodes, source='FLYWIRE', target='FAFB14raw', verbose=False)
        except:
            print(f'#W#cannot convert {s.id}')
            continue
        # s = navis.TreeNeuron.get_graph_nx(s)
        if prune == 'cell_body_fiber':
            s = prune_cell_body_fiber(s)
            _ = navis.strahler_index(s)
        elif prune == 'backbone_only':
            s = prune_to_backbone(s)
        mapped_skels.append(s)
    return mapped_skels


def filter_by_distance(skel, seg_skels, missing, segment_index_dict, segment_node_dict):
    """
    filter segments that accidentally mapped to the segment tree
    :param skel: segment tree to filter
    :param seg_skels: segments with skeleton
    :param missing: segments withot skeleton
    :param segment_node_dict: segments -> nodes
    """

    points = np.asarray(list(zip(skel.nodes['x'], skel.nodes['y'], skel.nodes['z'])))
    filter_id = []
    segment_distance_dict = {}
    # filter using google skeleton
    for s in seg_skels:
        if s.id == 4473590985:
            print('debug')
        if len(segment_index_dict[s.id]) > 40:
            continue
        nodes = s.vertices
        d = chamfer_distance(points * [4, 4, 40], nodes, metric='l2', direction='y_to_x')
        segment_distance_dict[s.id] = d
        if d > 2 * np.mean(skel.nodes['radius'][segment_index_dict[s.id]]) or \
                (len(nodes) > 6 and len(segment_index_dict[s.id]) < 2):
            filter_id.append(s.id)
        # if d > 1000:
        #     filter_id.append(int(s.id))
    filter_id.append(0)
    # filter using 20 sampled points
    # for m in tqdm(missing):
    #     if m < 1:
    #         filter_id.append(m)
    #     nodes = segment_node_dict[m]
    #     for n in nodes:
    #         seg_area = np.where(vol_ffn1[skel.nodes['x'][n] / 4 - 50: skel.nodes['x'][n] / 4 + 50,
    #                    skel.nodes['y'][n] / 4 - 50 : skel.nodes['y'][n] / 4 + 50,
    #                    skel.nodes['z'][n] - 2 : skel.nodes['z'][n] + 2] == m)
    #         sample = np.random.choice(len(seg_area[0]), np.minimum(20, len(seg_area[0])), replace=False)
    #         seg_points = np.asarray([seg_area[0][sample]*4 + skel.nodes['x'][n] - 200, seg_area[1][sample]*4 +
    #                                  skel.nodes['y'][n] - 200, seg_area[2][sample] + skel.nodes['z'][n] - 2]).transpose(1,0)
    #         d = chamfer_distance(points*[4, 4, 40], seg_points*[4, 4, 40], direction='y_to_x')
    #         print(d)
    #         if d > 1000:
    #             filter_id.append(int(m))
    #         segment_distance_dict[m] = d
    n_node_before_filter = skel.n_nodes
    background_nodes = segment_node_dict[0]
    for seg_id in filter_id:
        node_ids = segment_node_dict[seg_id]
        for node in node_ids:
            if skel.nodes['parent_id'][skel.nodes['node_id'] == node].item() == -1 \
                    and len(np.where(skel.nodes['parent_id'] == node)[0]) > 1:
                print(f'#I#Branching root node {node} removing')
                # pandas series: note difference between series.index and slice. np.where return slice, not index
                branch = skel.nodes['parent_id'][skel.nodes['parent_id'].values == node].index
                navis.reroot_skeleton(skel, skel.nodes['node_id'][branch[0]], inplace=True)
            navis.remove_nodes(skel, node, inplace=True)
        del segment_node_dict[seg_id]
    n_node_after_filter = skel.n_nodes
    print(
        f'#I#filtered {len(filter_id)} segments, removed {n_node_before_filter - n_node_after_filter} nodes ({len(background_nodes)} are background)')


def segment_weight(segment_node_dict):
    """
    :param segment_node_dict: segments -> nodes
    :return: segment_weight_dict: segments -> number of nodes
    """
    segment_weight_dict = {}
    for seg_id in segment_node_dict:
        segment_weight_dict[seg_id] = len(segment_node_dict[seg_id])
    return segment_weight_dict


def mapping_segments(skel, vol_ffn1):
    """
    :param skel: mapped flywire neuron skeleton
    :param vol_ffn1: segmentation with skeleton
    :return: a mapped tree neuron (NetworkX) with additional node property as mapped segment id
    """
    skel = skel.copy()
    segment_node_dict = {}
    segment_index_dict = {}
    skel.nodes['seg_id'] = np.zeros(skel.nodes['x'].shape)
    # save node_id because the index will change if reroot.
    for node in tqdm(range(skel.n_nodes)):
        seg_id = vol_ffn1[skel.nodes['x'][node] / 4, skel.nodes['y'][node] / 4, skel.nodes['z'][node]].item()
        skel.nodes['seg_id'][node] = seg_id
        if seg_id in segment_node_dict:
            segment_node_dict[seg_id].append(skel.nodes['node_id'][node])
            segment_index_dict[seg_id].append(node)
        else:
            segment_node_dict[seg_id] = [skel.nodes['node_id'][node]]
            segment_index_dict[seg_id] = [node]

    ids = np.unique(np.asarray(skel.nodes[:]['seg_id']))
    print(f'#I#Before filtering: {len(ids)}')
    seg_skels, missing = vol_ffn1.skeleton.get(ids)
    # print(missing)

    filter_by_distance(skel, seg_skels, missing, segment_index_dict, segment_node_dict)
    skel.segment_length = segment_weight(segment_node_dict)

    # replace the node with neibors
    # for n in segment_node_dict[s.id]:
    #     node_id = skel.nodes['node_id'][n]
    #     child_id = skel.nodes['node_id'][skel.nodes.parent_id == node_id]
    #     child_segid = skel.nodes['seg_id'][skel.nodes.node_id == child_id]
    #     parent_segid = skel.nodes['seg_id'][skel.nodes.parent_id == skel.nodes['parent_id'][n]]
    #
    #     skel.nodes['seg_id'][n]

    # filtered_ids = [k for k, v in distance_dict.items() if v < 30]
    # print(filtered_ids)
    return skel


def connection_weight(skel, node0, node1):
    """
    Compute connection length of the two directions by cutting the tree
    :param skel: mapped tree neuron
    :param edge: connector
    :return: connection length of the two directions
    """
    for subtree in skel.subtrees:
        if node0 and node1 in subtree:
            if navis.distal_to(skel, node0, node1):
                sub_tree0, sub_tree1 = navis.cut_skeleton(skel, node0)
                return sub_tree0.n_nodes, sub_tree1.n_nodes - 1
            else:
                sub_tree1, sub_tree0 = navis.cut_skeleton(skel, node1)
                return sub_tree0.n_nodes - 1, sub_tree1.n_nodes
    print(f'#E#cannot find {node0} in {skel.id}')


def get_connector(skel):
    """
    :param skel: mapped tree neuron
    :return: connector table, wrapped nodes filtered by connection record (all_connection)
    """
    connector_table = {'node0_id': [], 'node1_id': [], 'node0_cord': [], 'node1_cord': [], 'node0_segid': [],
                       'node1_segid': [], 'node0_weight': [], 'node1_weight': [], 'Strahler order': []}
    connector_table = pd.DataFrame(connector_table)
    all_connection = []
    for edge in skel.edges:
        node0 = skel.nodes['node_id'] == edge[0]
        node1 = skel.nodes['node_id'] == edge[1]
        all_connection.append([skel.nodes['seg_id'][node0].item(), skel.nodes['seg_id'][node1].item()])
    for edge in skel.edges:
        node0 = skel.nodes['node_id'] == edge[0]
        node1 = skel.nodes['node_id'] == edge[1]
        if skel.nodes['seg_id'][node0].item() != skel.nodes['seg_id'][node1].item():
            if [skel.nodes['seg_id'][node1].item(), skel.nodes['seg_id'][node0].item()] not in all_connection:
                # filter wrapped connector
                weight0, weight1 = connection_weight(skel, edge[0], edge[1])
                try:
                    connector_table.loc[len(connector_table.index)] = [edge[0], edge[1],
                                                                   [skel.nodes['x'][node0].item(),
                                                                    skel.nodes['y'][node0].item(),
                                                                    skel.nodes['z'][node0].item()],
                                                                   [skel.nodes['x'][node1].item(),
                                                                    skel.nodes['y'][node1].item(),
                                                                    skel.nodes['z'][node1].item()],
                                                                   skel.nodes['seg_id'][node0].item(),
                                                                   skel.nodes['seg_id'][node1].item(),
                                                                   weight0, weight1,
                                                                   skel.nodes.strahler_index[node1]]
                except:
                    print(f'#W#Maybe is not neuron!')
                    return None
    return connector_table


if __name__ == "__main__":
    create_logger(name='l1', file='/braindat/lab/liusl/flywire/log/flywire2fafbffn_debug3.log', sub_print=True, file_level='DEBUG')
    target_tree_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data'
    target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    visualization_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/visualization'

    vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True, parallel=True)  #
    vol_ffn1.parallel = 8
    vol_ffn1.meta.info['skeletons'] = 'skeletons_32nm'
    vol_ffn1.skeleton.meta.refresh_info()
    vol_ffn1.skeleton.meta.info['sharding']['hash'] = 'murmurhash3_x86_128'
    vol_ffn1.skeleton = ShardedPrecomputedSkeletonSource(vol_ffn1.skeleton.meta, vol_ffn1.cache, vol_ffn1.config)

    flywire_skel_path = '/braindat/lab/liusl/flywire/gt_skel'
    file_gt_skels = os.listdir(flywire_skel_path)
    random.shuffle(file_gt_skels)
    # get k neurons per iter
    for file_gt_skel in file_gt_skels:
        vol_ffn1.cache.flush()
        gt_skel = navis.read_json(os.path.join(flywire_skel_path, file_gt_skel))
        gt_skel = navis.read_json('/braindat/lab/liusl/flywire/gt_skel/720575940659093121.json')
        gt_skel = gt_skel[0]
        if gt_skel.n_nodes < 100:
            print(f'#W#{gt_skel.id} is not a neuron.')
            continue
        gt_skel.soma = None
        filename = os.path.join(target_tree_path, str(gt_skel.id) + '.json')
        if True:
            print(f'#I#building segment tree for {gt_skel.id}')
            start = time.time()
            mapped_skel = mapping_segments(gt_skel, vol_ffn1)
            map_time = time.time()
            print(f'#D#mapping time {map_time-start}')
            connector_table = get_connector(mapped_skel)
            con_time = time.time()
            print(f'#D#connector computing time {con_time - map_time}')
            filename_connector = os.path.join(target_connector_path, str(gt_skel.id) + '_connector.csv')
            navis.write_json(mapped_skel, filename, default=default_dump)
            if connector_table is not None:
                # check if connector contains only segments maintained after filtering (connectors build based on tree data)
                assert set(connector_table['node0_segid'].values.astype(int)) | \
                       set(connector_table['node1_segid'].values.astype(int)) < set(mapped_skel.segment_length.keys())
                connector_table.to_csv(filename_connector)
                visualize = pd.DataFrame(mapped_skel.segment_length, index=[0])
                visualize.to_csv(os.path.join(visualization_path, str(gt_skel.id) + '_save.csv'))
                print(f'#D#saving time {time.time() - con_time}')
