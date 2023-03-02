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


# def farthest_point_sample(filename, point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:,:3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     file=open(filename,'w')
#     file.writelines("ply\n")
#     file.writelines("format ascii 1.0\n")
#     file.writelines("element vertex "+str(511+1)+"\n")
#     file.writelines("property float x\n")
#     file.writelines("property float y\n")
#     file.writelines("property float z\n")
#     file.writelines("element face 0\n")
#     file.writelines("property list uchar int vertex_indices\n")
#     file.writelines("end_header\n")
#     for i in point:
#         file.writelines(str(float(i[0]))+"\t"+str(float(i[1]))+"\t"+str(float(i[2]))+"\n")
#
# def fps(id, filepath_data,filepath_output):
#     print("开始计时0")
#     time_start = time.time()
#     #filepath_data = '/braindat/lab/wangcx/proofreader-master/proofreader-master/proofreader/PC2/nagetive'
#     filenames = os.listdir(filepath_data)
#     n = len(filenames)
#     print(n)
#     #filepath_output = '/braindat/lab/wangcx/proofreader-master/proofreader-master/proofreader/PC2_512/nagetive'
#     for ii in range(0,n):
#         print(filenames[ii])
#         pnode = os.path.join(filepath_data,  filenames[ii])
#         out = os.path.join(filepath_output, filenames[ii])
#         if not os.path.exists(out):
#                 os.makedirs(out)
#         idnames = os.listdir(pnode)
#         N = len(idnames)
#         print(N)
#         for i in range(0,N):
#             filename =  idnames[i]
#             #print(filename)
#             pathin = os.path.join(pnode, filename)
#             pathout = os.path.join(out,  filename)
#             points = read_ply_points(pathin)
#             #print(points.shape)
#             if(points.shape[0]<=512):
#                 continue
#             farthest_point_sample(pathout, points, 512)
#     time_end = time.time()
#     time_c= time_end - time_start   #运行所花时间
#     print('time cost', time_c, 's')

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
        file.writelines(str(float(p[0])) + "\t" + str(float(p[1])) + "\t" + str(float(p[2])) + "\t" + str(int(id)) + "\t" + "\n")


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


if __name__ == '__main__':
    path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train/pos'
    dirs = os.listdir(path)
    N = 1024
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps/pos'
    target_h5_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_h5/pos'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    B = 128
    for dir in dirs:
        filepath_input = os.path.join(path, dir)
        t = time.time() - os.path.getmtime(filepath_input)
        if t > 60:
            filepath_output = os.path.join(target_path, dir)
            filepath_output_h5 = os.path.join(target_h5_path, dir)
            if not os.path.exists(filepath_output):
                os.makedirs(filepath_output)
            pcds = os.listdir(filepath_input)
            pcd_list = []
            ids_list = []
            pcd_down_list = []
            ids_down_list = []
            for pcd_name in pcds:
                pcd = read_ply_points(os.path.join(filepath_input, pcd_name))
                ids = PlyData.read(os.path.join(filepath_input, pcd_name)).elements[0].data['id']
                if len(np.unique(ids) > 1):
                    pcd_list.append(np.expand_dims(pcd, axis=0))
                    ids_list.append(ids)
            num = len(pcd_list)
            processed_batch = 0
            while processed_batch < num:
                pcd_B = pcd_list[processed_batch:min(processed_batch+B, len(pcd_list)-1)]
                ids_B = ids_list[processed_batch:min(processed_batch+B, len(pcd_list)-1)]
                pcd = np.concatenate(pcd_B, dim=0)
                pcd_tensor = torch.tensor(pcd, dtype=torch.long).unsqueeze(dim=0).to(device)
                pcd_down_index = farthest_point_sample(pcd_tensor, N)
                pcd_down = index_points(pcd_tensor, pcd_down_index)
                for k in range(pcd_down.sahpe[0]):
                    pcd_out = pcd_down[k, :, :]
                    ids_down = ids_B[k][pcd_down_index.cpu()][0]
                    ids_down_list.append(ids_down)
                processed_batch = processed_batch + B
            for i in range(len(pcd_down_list)):
                pcd_out = pcd_down_list[i]
                ids_down = ids_down_list
                write_ply(pcd_out, ids_down, os.path.join(filepath_output, pcd_name))
