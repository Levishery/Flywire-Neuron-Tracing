import math
import os
import numpy as np
import argparse
# /braindat/lab/liusl/flywire/block_data/30_percent
import pandas as pd
from tqdm import tqdm
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
    prefixes = ['train']
    for prefix in prefixes:
        file_name = f'/h3cstore_nt/JaneChen/SNEMI3D/edges-0nm-8x64x64/{prefix}ing/SNEMI3D-{prefix}-segmentation-wos-reduced-nodes-0-17x129x129-SNEMI3D.csv'

        pos_path_output = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}/pos'
        neg_path_output = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}/neg'
        sampled_pos_path_output = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/pos'
        sampled_neg_path_output = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/neg'
        # file_name = f'/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/edges-0nm-8x64x64/{prefix}ing/SNEMI3D-{prefix}-segmentation-wos-reduced-nodes-0-17x129x129-SNEMI3D.csv'
        #
        # pos_path_output = f'/braindat/lab/liusl/aaai24/snemi3d/pc-{prefix}/pos'
        # neg_path_output = f'/braindat/lab/liusl/aaai24/snemi3d/pc-{prefix}/neg'
        # sampled_pos_path_output = f'/braindat/lab/liusl/aaai24/snemi3d/pc-{prefix}-2048/pos'
        # sampled_neg_path_output = f'/braindat/lab/liusl/aaai24/snemi3d/pc-{prefix}-2048/neg'
        padding = np.asarray([160, 160, 32])
        os.makedirs(sampled_neg_path_output)
        os.makedirs(sampled_pos_path_output)
        lx = 160
        ly = 160
        lz = 32
        step = 1
        npoint = 2048
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        df = pd.read_csv(file_name)
        N = len(df)
        ii = 0

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

            filename1 = segid1
            filename2 = segid2
            # filename_pcd = os.path.join(filepath_output, str(filename1)+ str(filename2)+'1'+'.pcd')
            filename_pcd = os.path.join(filepath_output, str(filename1) +'_' + str(filename2) + '.ply')

            pcd = read_ply_points(filename_pcd)
            ids = PlyData.read(filename_pcd).elements[0].data['id']
            if len(np.unique(ids) > 1):
                pcd_tensor = torch.tensor(pcd, dtype=torch.long).unsqueeze(dim=0).to(device)
                pcd_down_index = farthest_point_sample(pcd_tensor, npoint)
                pcd_down = index_points(pcd_tensor, pcd_down_index)
                pcd_out = pcd_down[0, :, :]
                ids_down = ids[pcd_down_index.cpu()][0]
                write_ply(pcd_out, ids_down, os.path.join(sampled_output, str(filename1) +'_' + str(filename2) + '.ply'), npoint)
