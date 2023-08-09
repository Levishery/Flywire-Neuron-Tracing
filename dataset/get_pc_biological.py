import math
import os

import cv2
import cc3d
import numpy as np
# /braindat/lab/liusl/flywire/block_data/30_percent
import pandas as pd
from tqdm import tqdm
from utils import xpblock_to_fafb
from cloudvolume import CloudVolume
from fafbseg import google
import torch
from plyfile import PlyData
import open3d as o3d


def read_ply_points(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    # ply = PlyData.read(ply_path)
    # data = ply.elements[0].data
    # points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def select_points(bbox, points):
    [[bx1, by1, bz1], [bx2, by2, bz2]] = bbox
    ll = np.array([bx1, by1, bz1])  # lower-left
    ur = np.array([bx2, by2, bz2])  # upper-right

    inidx = np.all(np.logical_and(ll <= points, points <= ur), axis=1)
    return inidx


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = (torch.ones(B, N).to(device) * 1e10).long()
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def write_ply(point, ids, filename, N):
    file = open(filename, 'w')
    file.writelines("ply\n")
    file.writelines("format ascii 1.0\n")
    file.writelines("element vertex " + str(N) + "\n")
    file.writelines("property float x\n")
    file.writelines("property float y\n")
    file.writelines("property float z\n")
    file.writelines("property int id\n")
    file.writelines("element face 0\n")
    file.writelines("property list uchar int vertex_indices\n")
    file.writelines("end_header\n")
    for i in range(len(point)):
        p = point[i]
        id = ids[i]
        file.writelines(
            str(float(p[0])) + "\t" + str(float(p[1])) + "\t" + str(float(p[2])) + "\t" + str(int(id)) + "\t" + "\n")


def nearest_nonzero_idx(a,x,y,z):
    idx = np.argwhere(a)

    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x,y,z]).all(1)]

    return idx[((idx - [x,y,z])**2).sum(1).argmin()]



def findid3(df, index, offset):
    segid1 = df.iloc[index, 0]
    segid2 = df.iloc[index, 1]
    segid3 = segid2
    co = df.iloc[index, 2]
    co = co[1:-1]
    co = ' '.join(co.split())
    co = co.split(' ')
    x = float(co[0])
    y = float(co[1])
    z = float(co[2])
    i = 0
    while (segid3 == segid1 or segid3 == 0 or segid3 == segid2):
        i += 1
        locs = [x + offset * i, y + offset * i, z + offset / 4 * i]
        segid3 = google.locs_to_segments(locs, dataset='fafb-ffn1-20200412')
        segid3 = segid3[0]
        # segid3 = segid3[1:-1]
        if (i > 10):
            segid3 = -1
    return segid3


def get_fps(filepath_input, pcd_name, device):
    N = 2048
    lx = 80
    ly = 80
    lz = 32
    # crop_center not implemented, please set it the same as in get_pc()
    pcd = read_ply_points(os.path.join(filepath_input, pcd_name))
    ids = PlyData.read(os.path.join(filepath_input, pcd_name)).elements[0].data['id']
    if len(np.unique(ids)) > 1:
        pcd_tensor = torch.tensor(pcd, dtype=torch.long).unsqueeze(dim=0).to(device)
        pcd_down_index = farthest_point_sample(pcd_tensor, N)
        pcd_down = index_points(pcd_tensor, pcd_down_index)
        pcd_out = pcd_down[0, :, :]
        ids_down = ids[pcd_down_index.cpu()][0]
        write_ply(pcd_out, ids_down, os.path.join(filepath_input, pcd_name), N)


def work(split_df_index, split_total, file_name, filepath_data, path_output, device_id):
    lx = 80
    ly = 80
    lz = 32
    step = 1
    file_path = os.path.join(filepath_data, file_name)
    df_total = pd.read_csv(file_path, header=None)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    splits_df = np.array_split(df_total, split_total)
    df = splits_df[split_df_index]
    filepath_output = os.path.join(path_output, str(file_name)[:-4])

    vol = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True)
    # print(df)
    # #print(df.iloc[3,:])
    N = len(df)
    for ii in tqdm(range(0, N)):
        label = df.iloc[ii, 3]
        segid1 = df.iloc[ii, 0]
        segid2 = df.iloc[ii, 1]
        filename1 = segid1
        filename2 = segid2
        # filename_pcd = os.path.join(filepath_output, str(filename1)+ str(filename2)+'1'+'.pcd')
        filename_pcd = os.path.join(filepath_output, str(filename1) +'_' + str(filename2) + '.ply')
        if not os.path.exists(filename_pcd):
            cord = df.iloc[ii, 2][1:-1].split(' ')
            cord = [item for item in cord if item != ""]

            x,y,z= [float(cord[0]), float(cord[1]), float(cord[2])]
            vol_ = vol[int(x/4 - lx):int(x/4 + lx), int(y/4 - ly):int(y/4 + ly), int(z - lz):int(z + lz)]
            vol0 = np.where(vol_ == segid1, 255, vol_)
            vol0 = np.where(vol0 != 255, 0, vol0)
            # print(np.sum(vol0 == 255))
            vol0 = cc3d.connected_components(vol0[:,:,:,0], out_dtype=np.uint64)
            # print(np.unique(vol0))
            relabel0 = vol0[tuple(nearest_nonzero_idx(vol0,lx,ly,lz))]
            vol0 = np.where(vol0 == relabel0, 255, vol0)
            vol0 = np.where(vol0 != 255, 0, vol0)
            # print(np.sum(vol0 == 255))
            # print(np.sum(vol0 == 255))
            # vol1 = vol[int(x - lx):int(x + lx), int(y - ly):int(y + ly), int(z - lz):int(z + lz)]
            vol1 = np.where(vol_ == segid2, 255, vol_)
            vol1 = np.where(vol1 != 255, 0, vol1)
            # print(np.sum(vol1 == 255))
            vol1 = cc3d.connected_components(vol1[:,:,:,0], out_dtype=np.uint64)
            # print(np.unique(vol1))
            relabel1 = vol1[tuple(nearest_nonzero_idx(vol1,lx,ly,lz))]
            vol1 = np.where(vol1 == relabel1, 255, vol1)
            vol1 = np.where(vol1 != 255, 0, vol1)
            # print(np.sum(vol1 == 255))
            # print(np.sum(vol1 == 255))

            fid1 = open(filename_pcd, 'w')

            temp1 = 0
            for i in range(0, 2 * lz):
                data_tmp0 = vol0[:, :, i]
                data_tmp0 = data_tmp0.astype(np.uint8)
                contours0 = cv2.findContours(data_tmp0, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                boundary0 = np.array(contours0[0])
                boundary0 = [y for x in boundary0 for y in x]
                boundary0 = np.array(boundary0).squeeze()
                # print(i)
                if (len(boundary0.shape) == 1):
                    continue
                fid1.write(str((boundary0[0, 1]*16 + x*4 - lx*16)) + "\t" + str((boundary0[0, 0]*16 + y*4 - ly*16)))
                fid1.write("\t" + str((i + z - lz) * 40))
                fid1.write("\t" + str(0))
                fid1.write("\n")
                temp1 += 1
                # if(len(boundary0!=0)):
                #    print(i+z-lz)
                # step0 = int(len(boundary0)/10)+1
                # print(step0)
                k = 0
                for n in range(1, len(boundary0)):
                    p = [boundary0[k, 0], boundary0[k, 1]]
                    q = [boundary0[n, 0], boundary0[n, 1]]
                    d = math.dist(p, q)
                    if (d > step):
                        temp1 += 1
                        k = n
                        # print(d)
                        # fid1.write(str(boundary0[n, 0]) + " " + str(boundary0[n, 1]))
                        fid1.write(str((boundary0[n, 1]*16 + x*4 - lx*16)) + "\t" + str(boundary0[n, 0]*16 + y*4 - ly*16))
                        # fid.write(str(boundary[n]) + " " + str(boundary[n]))
                        fid1.write("\t" + str((i + z - lz) * 40))
                        fid1.write("\t" + str(0))
                        fid1.write("\n")
                    else:
                        continue

            for j in range(0, 2 * lz):
                data_tmp1 = vol1[:, :, j]
                data_tmp1 = data_tmp1.astype(np.uint8)
                contours1 = cv2.findContours(data_tmp1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                boundary1 = np.array(contours1[0])
                boundary1 = [y for x in boundary1 for y in x]
                boundary1 = np.array(boundary1).squeeze()
                # print(boundary1.shape)
                if (len(boundary1.shape) == 1):
                    continue
                fid1.write(str((boundary1[0, 1] * 16 + x * 4 - lx * 16)) + "\t" + str(
                    (boundary1[0, 0] * 16 + y * 4 - ly * 16)))
                fid1.write("\t" + str((j + z - lz) * 40))
                fid1.write("\t" + str(1))
                fid1.write("\n")
                temp1 += 1
                k = 0
                for n in range(1, len(boundary1)):
                    p = [boundary1[k, 0], boundary1[k, 1]]
                    q = [boundary1[n, 0], boundary1[n, 1]]
                    d = math.dist(p, q)
                    if (d > step):
                        temp1 += 1
                        k = n
                        # fid1.write(str(boundary1[n, 0]) + " " + str(boundary1[n, 1]))
                        # print(d)
                        fid1.write(str((boundary1[n, 1]*16 + x*4 - lx*16)) + "\t" + str(boundary1[n, 0]*16 + y*4 - ly*16))
                        # fid.write(str(boundary[n]) + " " + str(boundary[n]))
                        fid1.write("\t" + str((j + z - lz) * 40))
                        fid1.write("\t" + str(1))
                        fid1.write("\n")
                    else:
                        continue

            fid1.close()
            with open(filename_pcd, 'r+') as fid_:
                content = fid_.read()
                fid_.seek(0, 0)
                fid_.write(
                    'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty int id\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.format(
                        temp1) + content)
                fid_.close()
            get_fps(filepath_output, str(filename1) +'_' + str(filename2) + '.ply', device)

from functools import partial
import argparse
def get_args():
    parser = argparse.ArgumentParser(description="paths")
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--device_id', type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    filepath_data = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/'
    # file_name = 'evaluate_fafb_dust1200_dis500_rad0.3216_0.csv'
    file_name = 'evaluate_fafb_dust1200_dis500_rad0.3216_' + args.file_name

    file_path = os.path.join(filepath_data, file_name)

    torch.multiprocessing.set_start_method('spawn')
    path_output = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/biological/'
    filepath_output = os.path.join(path_output, str(file_name)[:-4])
    if not (os.path.exists(filepath_output)):
        os.makedirs(filepath_output)
    df = pd.read_csv(file_path, header=None)

    N = len(df)
    num_worker = 4
    num_worker_real = min(num_worker, os.cpu_count())
    split_total = num_worker * 10
    split_index_list = range(0, split_total)
    if num_worker_real > 0:
        partial_work = partial(work, file_name=file_name, filepath_data=filepath_data, path_output=path_output, split_total=split_total, device_id=args.device_id)
        with torch.multiprocessing.Pool(num_worker_real) as pool:
            for res in tqdm(
                    pool.imap_unordered(partial_work, split_index_list),
                    total=len(split_index_list),
            ):
                continue