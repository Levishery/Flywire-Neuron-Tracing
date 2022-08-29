# from cloudvolume import CloudVolume
# import cloudvolume
# from tqdm import tqdm
# import os
# from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# from tqdm import tqdm
# import urllib
#
# def chamfer_distance(x, y, metric='l2', direction='bi'):
#     """Chamfer distance between two point clouds
#     Parameters
#     ----------
#     x: numpy array [n_points_x, n_dims]
#         first point cloud
#     y: numpy array [n_points_y, n_dims]
#         second point cloud
#     metric: string or callable, default ‘l2’
#         metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
#     direction: str
#         direction of Chamfer distance.
#             'y_to_x':  computes average minimal distance from every point in y to x
#             'x_to_y':  computes average minimal distance from every point in x to y
#             'bi': compute both
#     Returns
#     -------
#     chamfer_dist: float
#         computed bidirectional Chamfer distance:
#             sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
#     """
#
#     if direction == 'y_to_x':
#         x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
#         min_y_to_x = x_nn.kneighbors(y)[0]
#         chamfer_dist = np.mean(min_y_to_x)
#     elif direction == 'x_to_y':
#         y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
#         min_x_to_y = y_nn.kneighbors(x)[0]
#         chamfer_dist = np.mean(min_x_to_y)
#     elif direction == 'bi':
#         x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
#         min_y_to_x = x_nn.kneighbors(y)[0]
#         y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
#         min_x_to_y = y_nn.kneighbors(x)[0]
#         chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
#     else:
#         raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
#
#     return chamfer_dist
#
# point_path = '/Users/janechen/Desktop/实验室/untitled/flywire_neuroskel/mapped_points'
# cv = CloudVolume('precomputed://gs://fafb-ffn1-20190805/segmentation', use_https=True, parallel=True)
# cv.parallel = 4
# cv.meta.info['skeletons'] = 'skeletons_32nm'
# cv.skeleton.meta.refresh_info()
# cv.skeleton.meta.info['sharding']['hash']='murmurhash3_x86_128'
# cv.skeleton = ShardedPrecomputedSkeletonSource(cv.skeleton.meta, cv.cache, cv.config)
# skel16_url = 'https://storage.googleapis.com/fafb-ffn1/fafb-public-skeletons/16'
# with urllib.request.urlopen(skel16_url) as source:
#   skel_test = cloudvolume.Skeleton.from_precomputed(source.read(), segid=424295)
#
# vol = CloudVolume('file:///Volumes/xcy_braindata/braindata/google_segmentation/google_16.0x16.0x40.0', mip = 0, parallel = True, cache=True)
# point_files = os.listdir(point_path)
#
# points = skel_test.vertices/[4,4,40]
# ids = []
# id_dict = {}
# for pos in tqdm(points):
#     id = vol[pos[0] / 4, pos[1] / 4, pos[2]]
#     id_dict[id.item()] = pos
#     ids.append(id.item())
# ids = np.unique(np.asarray(ids))
# print('Before filtering: ', len(ids))
# skels, missing = cv.skeleton.get(ids)
# print(missing)
# distance_dict = {}
# for skel in skels:
#     nodes = skel.vertices
#     d = chamfer_distance(points, nodes/[4,4,40], direction='y_to_x')
#     print(skel.id, d)
#     distance_dict[skel.id] = d
# filtered_ids = [k for k, v in distance_dict.items() if v < 100]
# print(filtered_ids)
#
# # for file in point_files[20:]:
# #     points = np.load(os.path.join(point_path, file))
# #     ids = []
# #     id_dict = {}
# #     for pos in tqdm(points):
# #         id = vol[pos[0] / 4, pos[1] / 4, pos[2]]
# #         id_dict[id.item()] = pos
# #         ids.append(id.item())
# #     ids = np.unique(np.asarray(ids))
# #     print('Before filtering: ', len(ids))
# #     skels, missing = cv.skeleton.get(ids)
# #     print(missing)
# #     distance_dict = {}
# #     for skel in skels:
# #         nodes = skel.vertices
# #         d = chamfer_distance(points, nodes/[4,4,40], direction='y_to_x')
# #         print(skel.id, d)
# #         distance_dict[skel.id] = d
# #     filtered_ids = [k for k, v in distance_dict.items() if v < 30]
# #     print(filtered_ids)

import matplotlib.pyplot as plt
import fafbseg
import navis
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from cloudvolume import CloudVolume
from sklearn.neighbors import NearestNeighbors
from cilog import create_logger
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
import pickle
import os
import signal
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


def prune_cell_body_fiber(x, method = 'betweenness', reroot_soma: bool = True, heal: bool = True,
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
    try:
        subts = navis.cut_skeleton(x, cut)
        s = subts[0]
        _ = navis.strahler_index(s)
    except:
        print(f'#W# cannot prune {x.id}')
        return None
    return s


def prune_to_backbone(x):
    x = prune_cell_body_fiber(x)
    _ = navis.strahler_index(x)
    x = x.prune_by_strahler(to_prune=range(1, x.nodes.strahler_index.max()))
    return x

def get_mapped_flywire_neuron(neuron_id, prune = 'cell_body_fiber'):
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
            print(f'#W#cannot convert {s.name}')
            continue
        # s = navis.TreeNeuron.get_graph_nx(s)
        if prune == 'cell_body_fiber':
            s = prune_cell_body_fiber(s)
        elif prune == 'backbone_only':
            s = prune_to_backbone(s)
        mapped_skels.append(s)
    return mapped_skels


class TimeOutException(Exception):
    pass


def handle(signum, frame):
    raise TimeOutException("运行超时！")


def set_timeout(timeout, callback):
    def wrapper(func):
        def inner(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(timeout)  # 开启闹钟信号
                rs = func(*args, **kwargs)
                signal.alarm(0)  # 关闭闹钟信号
                return rs
            except TimeOutException as e:
                callback()

        return inner

    return wrapper


def process_time():
    print("超时了")


@set_timeout(50, process_time)
def get_mapped_flywire_neuron_single(neuron_id, prune = 'cell_body_fiber'):
    """
    :param neuron_id:
    :return: mapped tree neurons of neuron_id, from flywire to fafbv14
    """
    # swc_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/swc'
    # mapped_point_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/mapped_points'
    try:
        s = fafbseg.flywire.skeletonize_neuron(neuron_id)
    except:
        print(f'#W#cannot get {neuron_id}')
        return None
    # s.to_swc(os.path.join(swc_path, str(neuron_id) + '.swc'))
    if s.soma is None:
        print(f'#W#neuron {s.id} has no soma record, skip.')
        return 'no_soma'
    try:
        s.nodes = navis.xform_brain(s.nodes, source='FLYWIRE', target='FAFB14raw', verbose=False)
    except:
        print(f'#W#cannot convert {s.name}')
        return None
    # s = navis.TreeNeuron.get_graph_nx(s)
    if prune == 'cell_body_fiber':
        s = prune_cell_body_fiber(s)
    elif prune == 'backbone_only':
        s = prune_to_backbone(s)
    return s

def filter_by_distance(skel, seg_skels, missing, segment_node_dict):
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
        if len(segment_node_dict[s.id])>10:
            continue
        nodes = s.vertices
        d = chamfer_distance(points*[4, 4, 40], nodes, metric='l2', direction='y_to_x')
        segment_distance_dict[s.id] = d
        if d > 2*np.mean(skel.nodes['radius'][segment_node_dict[s.id]]):
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


    for seg_id in filter_id:
        navis.remove_nodes(skel, skel.nodes['node_id'][segment_node_dict[seg_id]], inplace=True)


def mapping_segments(skel, vol_ffn1, vol_ffn1_skel):
    """
    :param skel: mapped flywire neuron skeleton
    :param vol_ffn1: segmentation with skeleton
    :return: a mapped tree neuron (NetworkX) with additional node property as mapped segment id
    """
    skel = skel.copy()
    segment_node_dict = {}
    skel.nodes['seg_id'] = np.zeros(skel.nodes['x'].shape)
    for node in tqdm(range(skel.n_nodes)):
        seg_id = vol_ffn1[skel.nodes['x'][node] / 4, skel.nodes['y'][node] / 4, skel.nodes['z'][node]].item()
        skel.nodes['seg_id'][node] = seg_id
        if seg_id in segment_node_dict:
            segment_node_dict[seg_id].append(node)
        else:
            segment_node_dict[seg_id] = [node]

    ids = np.unique(np.asarray(skel.nodes[:]['seg_id']))
    print(f'#D#Before filtering: {len(ids)}')
    seg_skels, missing = vol_ffn1_skel.skeleton.get(ids)
    # print(missing)

    filter_by_distance(skel, seg_skels, missing, segment_node_dict)
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


def get_connector(skel):
    """

    :param skel: mapped tree neuron
    :return: connector table, wrapped nodes filtered
    """
    connector_table = {'node1_id':[], 'node2_id':[], 'node1_cord':[], 'node2_cord':[],  'node1_segid':[], 'node2_segid':[], 'Strahler order':[]}
    connector_table = pd.DataFrame(connector_table)
    all_connection = []
    for edge in skel.edges:
        node1 = skel.nodes['node_id'] == edge[0]
        node2 = skel.nodes['node_id'] == edge[1]
        all_connection.append([skel.nodes['seg_id'][node1].item(), skel.nodes['seg_id'][node2].item()])
    for edge in skel.edges:
        node1 = skel.nodes['node_id'] == edge[0]
        node2 = skel.nodes['node_id'] == edge[1]
        if skel.nodes['seg_id'][node1].item() != skel.nodes['seg_id'][node2].item():
            if [skel.nodes['seg_id'][node2].item(), skel.nodes['seg_id'][node1].item()] not in all_connection:
                # filter wrapped connector
                connector_table.loc[len(connector_table.index)] = [edge[0], edge[1],
                    [skel.nodes['x'][node1].item(), skel.nodes['y'][node1].item(), skel.nodes['z'][node1].item()], [skel.nodes['x'][node2].item(),
                        skel.nodes['y'][node2].item(), skel.nodes['z'][node2].item()], skel.nodes['seg_id'][node1].item(), skel.nodes['seg_id'][node2].item(), skel.nodes.strahler_index[node2]]
    return connector_table


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__=="__main__":
    create_logger(name='l1', file='/flywire_data//flywire2fafbffn1.log', sub_print=True)
    fafbseg.flywire.set_chunkedgraph_secret("5814c407f6cca6b2c1e50f5f99e6a555")
    proof_list = pd.read_csv('/flywire_data/proof_stat_df.csv')
    download_stat = np.loadtxt('/flywire_data/dowload_stat.txt')
    target_path = '/flywire_data/gt_skel'

    # get k neurons per iter
    # k = 40
    # for i in range(0, len(proof_list), k):
    #     neuron_id = proof_list['pt_root_id'][i: i+k]
    #     print(f'Getting neurons: {neuron_id}')
    #     gt_skels = get_mapped_flywire_neuron(neuron_id)
    #     for skel in gt_skels:
    #         filename = os.path.join(target_path, str(skel.id) + '.json')
    #         if not os.path.exists(filename):
    #             navis.write_json(skel, filename, default=default_dump)
    k = 0
    for i in range(len(proof_list)):
        neuron_id = proof_list['pt_root_id'][i]
        if neuron_id not in download_stat:
            print(f'{k}: Getting neuron: {neuron_id}')
            gt_skel = get_mapped_flywire_neuron_single(neuron_id)
            k = k+1
            if gt_skel is not None:
                download_stat = np.append(download_stat, neuron_id)
                np.savetxt('/flywire_data/dowload_stat.txt', download_stat)
                if gt_skel != 'no_soma':
                    filename = os.path.join(target_path, str(gt_skel.id) + '.json')
                    navis.write_json(gt_skel, filename, default=default_dump)

