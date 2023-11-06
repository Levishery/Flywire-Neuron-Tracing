import math
import os

import cv2
import cc3d
import numpy as np
import argparse
# /braindat/lab/liusl/flywire/block_data/30_percent
import pandas as pd
import shutil
from utils import readh5
from tqdm import tqdm
from utils import xpblock_to_fafb
from cloudvolume import CloudVolume
from fafbseg import google
import torch
import open3d as o3d
from plyfile import PlyData


def nearest_nonzero_idx(a,x,y,z):
    idx = np.argwhere(a)

    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x,y,z]).all(1)]

    return idx[((idx - [x,y,z])**2).sum(1).argmin()]



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


def check_boundary(x, y, z, shape):
    [h, w, d] = shape
    is_x_bound = x == 0 or x == h-1
    is_y_bound = y == 0 or y == w-1
    is_z_bound = z == 0 or z == d-1
    return is_x_bound or is_y_bound or is_z_bound


def write_ply(point, ids, filename, npoint):
    file = open(filename, 'w')
    file.writelines("ply\n")
    file.writelines("format ascii 1.0\n")
    file.writelines("element vertex " + str(npoint) + "\n")
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
        file.writelines(str(float(p[0])) + "\t" + str(float(p[1])) + "\t" + str(float(p[2])) + "\t" + str(int(id)) + "\t" + "\n")


# filepath_data = r"E:\Desktop\aw"
# '/braindat/lab/liusl/flywire/block_data/train_30'
# filepath_data = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train_1000_reformat'
def get_args():
    parser = argparse.ArgumentParser(description="paths")
    parser.add_argument('--prefix', type=str, default='None')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    prefixes = ['train', 'test']
    for prefix in prefixes:
        file_name = f'/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/edges-0nm-8x64x64/{prefix}ing/SNEMI3D-{prefix}-segmentation-wos-reduced-nodes-0-17x129x129-SNEMI3D.csv'
        segmentation_path = f'/braindat/lab/liuyixiong/data/biologicalgraphs/biologicalgraphs/neuronseg/segmentations/SNEMI3D-{prefix}-segmentation-wos-reduced-nodes-400nm-3x20x60x60-SNEMI3D.h5'

        vol = readh5(segmentation_path)
        vol = np.transpose(vol, [2, 1, 0])
        h, w, d = vol.shape
        pos_path_output = f'/braindat/lab/liusl/aaai24/snemi3d/pc-{prefix}/pos'
        neg_path_output = f'/braindat/lab/liusl/aaai24/snemi3d/pc-{prefix}/neg'
        sampled_pos_path_output = f'/braindat/lab/liusl/aaai24/snemi3d/pc-{prefix}-2048/pos'
        sampled_neg_path_output = f'/braindat/lab/liusl/aaai24/snemi3d/pc-{prefix}-2048/neg'
        padding = np.asarray([160, 160, 32])
        vol_padding = np.zeros(vol.shape + padding * 2)
        vol_padding[padding[0]: h + padding[0], padding[1]: w + padding[1], padding[2]: d + padding[2]] = vol
        lx = 160
        ly = 160
        lz = 32
        step = 1
        npoint = 2048
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        df = pd.read_csv(file_name)
        N = len(df)
        ii = 0

        if os.path.exists(neg_path_output):
            shutil.rmtree(neg_path_output)
            shutil.rmtree(sampled_neg_path_output)
        if os.path.exists(pos_path_output):
            shutil.rmtree(pos_path_output)
            shutil.rmtree(sampled_pos_path_output)
        os.makedirs(neg_path_output)
        os.makedirs(sampled_neg_path_output)
        os.makedirs(pos_path_output)
        os.makedirs(sampled_pos_path_output)
        for ii in tqdm(range(0, N)):
            label = df.iloc[ii, 5]
            if label == 1:
                filepath_output = pos_path_output
                sampled_output = sampled_pos_path_output
            elif label == 0:
                filepath_output = neg_path_output
                sampled_output = sampled_neg_path_output
            else:
                continue
            segid1 = df.iloc[ii, 1]
            segid2 = df.iloc[ii, 2]
            # if segid1 == 6978865582:
            #     print('@')

            # cord = df.iloc[ii, 2][1:-1].split()
            # cord = [int(cord[0]), int(cord[1]), int(cord[2]), int(cord[3])]
            # cord = np.asarray(cord) + np.asarray([0, 8, 64, 64]) + np.asarray([0, 16+28, 156, 156])
            # fafb_cord = xpblock_to_fafb()

            x, y, z= [int(df.iloc[ii, 6]*2), int(df.iloc[ii, 7]*2), int(df.iloc[ii, 8])]
            vol_ = vol_padding[int(x + padding[0] - lx):int(x + padding[0] + lx)+1,
                   int(y + padding[1] - ly):int(y + padding[1] + ly)+1,
                   int(z + padding[2] - lz):int(z + padding[2] + lz)+1]
            vol0 = np.where(vol_ == segid1, 255, vol_)
            vol0 = np.where(vol0 != 255, 0, vol0)
            # print(np.sum(vol0 == 255))
            vol0 = cc3d.connected_components(vol0, out_dtype=np.uint64)
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
            vol1 = cc3d.connected_components(vol1, out_dtype=np.uint64)
            # print(np.unique(vol1))
            relabel1 = vol1[tuple(nearest_nonzero_idx(vol1,lx,ly,lz))]
            vol1 = np.where(vol1 == relabel1, 255, vol1)
            vol1 = np.where(vol1 != 255, 0, vol1)
            # print(np.sum(vol1 == 255))
            # print(np.sum(vol1 == 255))

            filename1 = segid1
            filename2 = segid2
            # filename_pcd = os.path.join(filepath_output, str(filename1)+ str(filename2)+'1'+'.pcd')
            filename_pcd = os.path.join(filepath_output, str(filename1) +'_' + str(filename2) + '.ply')
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
                # remove padding doundary
                fid1.write(str((boundary0[0, 1]*6)) + "\t" + str((boundary0[0, 0]*6)))
                fid1.write("\t" + str((i) * 30))
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
                        k = n
                        # print(d)
                        # fid1.write(str(boundary0[n, 0]) + " " + str(boundary0[n, 1]))
                        # remove padding doundary
                        fid1.write(str((boundary0[n, 1]*6)) + "\t" + str(boundary0[n, 0]*6))
                        # fid.write(str(boundary[n]) + " " + str(boundary[n]))
                        fid1.write("\t" + str((i) * 30))
                        fid1.write("\t" + str(0))
                        fid1.write("\n")
                        temp1 += 1
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
                fid1.write(str((boundary1[0, 1] * 6)) + "\t" + str(
                    (boundary1[0, 0] * 6)))
                fid1.write("\t" + str((j) * 30))
                fid1.write("\t" + str(1))
                fid1.write("\n")
                temp1 += 1
                k = 0
                for n in range(1, len(boundary1)):
                    p = [boundary1[k, 0], boundary1[k, 1]]
                    q = [boundary1[n, 0], boundary1[n, 1]]
                    d = math.dist(p, q)
                    if (d > step):
                        k = n
                        # fid1.write(str(boundary1[n, 0]) + " " + str(boundary1[n, 1]))
                        # print(d)
                        fid1.write(str((boundary1[n, 1]*6)) + "\t" + str(boundary1[n, 0]*6))
                        # fid.write(str(boundary[n]) + " " + str(boundary[n]))
                        fid1.write("\t" + str((j) * 30))
                        fid1.write("\t" + str(1))
                        fid1.write("\n")
                        temp1 += 1
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

            # downsample
            pcd = read_ply_points(filename_pcd)
            ids = PlyData.read(filename_pcd).elements[0].data['id']
            if len(np.unique(ids) > 1):
                pcd_tensor = torch.tensor(pcd, dtype=torch.long).unsqueeze(dim=0).to(device)
                pcd_down_index = farthest_point_sample(pcd_tensor, npoint)
                pcd_down = index_points(pcd_tensor, pcd_down_index)
                pcd_out = pcd_down[0, :, :]
                ids_down = ids[pcd_down_index.cpu()][0]
                write_ply(pcd_out, ids_down, os.path.join(sampled_output, str(filename1) +'_' + str(filename2) + '.ply'), npoint)
