from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
from cloudvolume import CloudVolume
import os
import psutil
import resource
from tqdm import tqdm
import pandas as pd
import navis
import trimesh as tm
import numpy as np
import skeletor as sk
from cloudvolume import Skeleton

p = psutil.Process()
print(p.pid)


def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))


limit_memory(1024*1024*1024*47)


def stat_mesh(vol_ffn1):
    block_connector = '/braindat/lab/liusl/flywire/block_data/train_30'
    csv_list = os.listdir(block_connector)
    vertices_list = []
    for f in tqdm(csv_list):
        vol_ffn1.cache.flush()
        f_name = os.path.join(block_connector, f)
        df = pd.read_csv(f_name, header=None)
        for i in df.index:
            seg = df.iloc[i][0]
            mesh = vol_ffn1.mesh.get(seg)
            vertices_list.append(len(mesh[seg].vertices))
            len1 = len(mesh[seg].vertices)
            if len(mesh[seg].vertices)<30:
                print(seg)
            seg = df.iloc[i][1]
            mesh = vol_ffn1.mesh.get(seg)
            vertices_list.append(len(mesh[seg].vertices))
            len2 = len(mesh[seg].vertices)
            if len2>len1: print('T')
            else: print('F')
            if len(mesh[seg].vertices)<30:
                print(seg)



vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True)  #
vol_ffn1.parallel = 8
vol_ffn1.meta.info['skeletons'] = 'skeletons_32nm'
vol_ffn1.skeleton.meta.refresh_info()
vol_ffn1.skeleton.meta.info['sharding']['hash'] = 'murmurhash3_x86_128'
vol_ffn1.skeleton = ShardedPrecomputedSkeletonSource(vol_ffn1.skeleton.meta, vol_ffn1.cache, vol_ffn1.config)

vol_ffn1_fullskel = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True)  #
vol_ffn1_fullskel.parallel = 8

radius = [80,80,16]

# stat_mesh(vol_ffn1)
block_connector = '/braindat/lab/liusl/flywire/block_data/train_30'
target_path = '/braindat/lab/liusl/flywire/block_data/skel_connector_data/train'
csv_list = os.listdir(block_connector)
block_done = {}
if os.path.exists('/code/dataset/block_done.npy'):
    block_done = np.load('/code/dataset/block_done.npy', allow_pickle=True).item()
for f in csv_list:
    if not (f in block_done and block_done[f]):
        skel_dir = os.path.join('/braindat/lab/lizl/google/google_16.0x16.0x40.0', 'skeletons_'+ f.split('.')[0])
        if not os.path.exists(skel_dir):
            os.makedirs(skel_dir)
        print('begin skeletonizing block', f)
        done_list = []
        vol_ffn1_fullskel.meta.info['skeletons'] = 'skeletons_'+ f.split('.')[0]
        vol_ffn1_fullskel.skeleton.meta.refresh_info()
        vol_ffn1.cache.flush()
        f_name = os.path.join(block_connector, f)
        df = pd.read_csv(f_name, header=None)
        df_new = df.copy()
        for i in tqdm(df.index):

            # put the larger segment (seg_start) in column 1
            # seg0 = df.iloc[i][0]
            # seg1 = df.iloc[i][1]
            # mesh = vol_ffn1.mesh.get([seg0, seg1])
            # len0 = len(mesh[seg0].vertices)
            # mesh = vol_ffn1.mesh.get(seg1)
            # len1 = len(mesh[seg1].vertices)
            # if not len1 > len0:
            #     df_new.iloc[i,0] = seg1
            #     df_new.iloc[i,1] = seg0

            loc = df.iloc[i][2][1:-1].split()
            cord = [float(loc[0]), float(loc[1]), float(loc[2])]
            ffn1_vol = vol_ffn1[cord[0]/4-radius[0]:cord[0]/4+radius[0]+1, cord[1]/4-radius[1]:cord[1]/4+radius[1]+1, cord[2]-radius[2]:cord[2]+radius[2]+1]
            segments = np.setdiff1d(np.unique(ffn1_vol), [0])
            del ffn1_vol
            skels, missing = vol_ffn1.skeleton.get(segments)
            del skels
            mesh_missing = vol_ffn1.mesh.get(missing)
            for id in missing:
                mesh = mesh_missing[id]
                if len(mesh.vertices) > 15 and not (id in done_list):
                    try:
                        # fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=True)
                        skel = sk.skeletonize.by_wavefront(mesh, waves=1, step_size=2, progress=False)
                        skel.save_swc('save.swc')
                        with open('save.swc', 'r') as f_swc:
                            x = f_swc.read()
                        skel = Skeleton.from_swc(x)
                        skel.id = id
                        vol_ffn1_fullskel.skeleton.upload(skel)
                        done_list.append(id)
                    except:
                        print('cannot skeletonize ', id)
        df_new.to_csv(os.path.join(target_path, f), header=False, index=False)
        block_done[f] = True
        np.save('/code/dataset/block_done.npy', block_done)


