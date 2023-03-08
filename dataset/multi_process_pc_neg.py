# FPS+H5
import numpy as np
import open3d as o3d
import torch
import random
import pandas as pd
import time
import os
from plyfile import PlyData
import h5py
import csv
from tqdm import tqdm
import multiprocessing


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


def write_ply(point, ids, filename):
    file = open(filename, 'w')
    file.writelines("ply\n")
    file.writelines("format ascii 1.0\n")
    file.writelines("element vertex " + str(1024) + "\n")
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


def h5(filepath_output, outpath):
    filenames = os.listdir(filepath_output)
    n = len(filenames)
    print(n)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # filepath_output = '/braindat/lab/wangcx/proofreader-master/proofreader-master/proofreader/PC2_512/nagetive'
    for ii in range(0, n):
        print(filenames[ii])
        pnode = os.path.join(filepath_output, filenames[ii])
        out = os.path.join(outpath, 'test%s.h5' % ii)
        idnames = os.listdir(pnode)
        N = len(idnames)
        f1 = open("%s/7495309906_h5.csv" % outpath, 'a', newline='')
        writer = csv.writer(f1)
        writer.writerow(idnames)
        f1.close()
        label = np.zeros((N, 1))
        data = np.zeros((N, 512, 3))
        print(N)
        for i in range(0, N):
            filename = idnames[i]
            path = os.path.join(pnode, filename)
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points)  # (512,3)
            data[i] = points
        f = h5py.File(out, 'w')  # 创建一个h5文件，文件指针是f
        f['data'] = data  # 将数据写入文件的主键data下面
        f['label'] = label  # 将数据写入文件的主键labels下面
        f.close()


def work(dir):
    N = 1024
    lx = 60
    ly = 60
    lz = 24
    crop_center = False
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test/neg'
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps/neg'
    filepath_data = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train_1000_reformat'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    filepath_input = os.path.join(path, dir)
    t = time.time() - os.path.getmtime(filepath_input)
    if t > 60:
        filepath_output = os.path.join(target_path, dir)
        # filepath_output_h5 = os.path.join(target_h5_path, dir)
        if not os.path.exists(filepath_output):
            os.makedirs(filepath_output)
            csv_path = os.path.join(filepath_data, str(dir)+'.csv')
            # if os.path.exists(csv_path):
            pcds = os.listdir(filepath_input)
            for pcd_name in tqdm(pcds):
                pcd = read_ply_points(os.path.join(filepath_input, pcd_name))
                seg_ids = pcd_name.split('.')[0].split('_')
                seg_ids = [int(seg_ids[0]), int(seg_ids[1])]
                ids = PlyData.read(os.path.join(filepath_input, pcd_name)).elements[0].data['id']
                if len(np.unique(ids)) > 1:
                    if crop_center:
                        df = pd.read_csv(csv_path, header=None)
                        center_cord = df.loc[(df[0] == seg_ids[0]) & (df[1] == seg_ids[1]), 5].values[0][1:-1].split(',')
                        center_cord = [float(center_cord[0]), float(center_cord[1]), float(center_cord[2])]
                        cords = pcd / np.asarray([4, 4, 40])
                        bbox = [list(center_cord-np.asarray([lx,ly,lz])* [4, 4, 1]), list(center_cord+np.asarray([lx,ly,lz])* [4, 4, 1])]
                        in_box_indexes = np.where(select_points(bbox, cords))[0]
                        if len(in_box_indexes) > len(ids)*0.4:
                            pcd = pcd[in_box_indexes]
                            ids = ids[in_box_indexes]
                            pcd_tensor = torch.tensor(pcd, dtype=torch.long).unsqueeze(dim=0).to(device)
                            pcd_down_index = farthest_point_sample(pcd_tensor, N)
                            pcd_down = index_points(pcd_tensor, pcd_down_index)
                            pcd_out = pcd_down[0, :, :]
                            ids_down = ids[pcd_down_index.cpu()][0]
                            write_ply(pcd_out, ids_down, os.path.join(filepath_output, pcd_name))
                    else:
                        pcd_tensor = torch.tensor(pcd, dtype=torch.long).unsqueeze(dim=0).to(device)
                        pcd_down_index = farthest_point_sample(pcd_tensor, N)
                        pcd_down = index_points(pcd_tensor, pcd_down_index)
                        pcd_out = pcd_down[0, :, :]
                        ids_down = ids[pcd_down_index.cpu()][0]
                        write_ply(pcd_out, ids_down, os.path.join(filepath_output, pcd_name))
        else:
            print('%s exists' % filepath_output)


if __name__ == '__main__':
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test/neg'
    dirs = os.listdir(path)
    torch.multiprocessing.set_start_method('spawn')

    num_worker = 4
    num_worker_real = min(num_worker, os.cpu_count())
    if num_worker_real > 0:
        with torch.multiprocessing.Pool(num_worker_real) as pool:
            for res in tqdm(
                    pool.imap_unordered(work, dirs),
                    total=len(dirs),
            ):
                continue