#FPS+H5
import numpy as np
import open3d as o3d
import random
import pandas as pd
import time
import os
import h5py
import csv


def read_ply_points(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    #ply = PlyData.read(ply_path)
    #data = ply.elements[0].data
    #points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points

def farthest_point_sample(filename, point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    file=open(filename,'w')
    file.writelines("ply\n")
    file.writelines("format ascii 1.0\n")
    file.writelines("element vertex "+str(511+1)+"\n")
    file.writelines("property float x\n")
    file.writelines("property float y\n")
    file.writelines("property float z\n")
    file.writelines("element face 0\n")
    file.writelines("property list uchar int vertex_indices\n")
    file.writelines("end_header\n")
    for i in point:
        file.writelines(str(float(i[0]))+"\t"+str(float(i[1]))+"\t"+str(float(i[2]))+"\n")

def fps(id, filepath_data,filepath_output):
    print("开始计时0")
    time_start = time.time()
    #filepath_data = '/braindat/lab/wangcx/proofreader-master/proofreader-master/proofreader/PC2/nagetive'
    filenames = os.listdir(filepath_data)
    n = len(filenames)
    print(n)
    #filepath_output = '/braindat/lab/wangcx/proofreader-master/proofreader-master/proofreader/PC2_512/nagetive'
    for ii in range(0,n):
        print(filenames[ii])
        pnode = os.path.join(filepath_data,  filenames[ii])
        out = os.path.join(filepath_output, filenames[ii])
        if not os.path.exists(out):
                os.makedirs(out)
        idnames = os.listdir(pnode)
        N = len(idnames)
        print(N)
        for i in range(0,N):
            filename =  idnames[i]
            #print(filename)
            pathin = os.path.join(pnode, filename)
            pathout = os.path.join(out,  filename)
            points = read_ply_points(pathin)
            #print(points.shape)
            if(points.shape[0]<=512):
                continue
            farthest_point_sample(pathout, points, 512)
    time_end = time.time()
    time_c= time_end - time_start   #运行所花时间
    print('time cost', time_c, 's')

def h5(filepath_output, outpath):
    filenames = os.listdir(filepath_output)
    n = len(filenames)
    print(n)
    if not os.path.exists(outpath):
                os.makedirs(outpath)
    #filepath_output = '/braindat/lab/wangcx/proofreader-master/proofreader-master/proofreader/PC2_512/nagetive'
    for ii in range(0,n):
        print(filenames[ii])
        pnode = os.path.join(filepath_output,  filenames[ii])
        out = os.path.join(outpath,'test%s.h5'%ii)
        idnames = os.listdir(pnode)
        N = len(idnames)
        f1 = open("%s/7495309906_h5.csv"%outpath, 'a',newline='')
        writer = csv.writer(f1)
        writer.writerow(idnames)
        f1.close()
        label = np.zeros((N, 1))
        data = np.zeros((N, 512,3))
        print(N)
        for i in range(0,N):
            filename =  idnames[i]
            path = os.path.join(pnode, filename)
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points)  # (512,3)
            data[i] = points
        f = h5py.File(out, 'w')  # 创建一个h5文件，文件指针是f
        f['data'] = data  # 将数据写入文件的主键data下面
        f['label'] = label  # 将数据写入文件的主键labels下面
        f.close()


if __name__ == '__main__':
    id = 7495309906
    filepath_data = 'result/7495309906/pc'
    filepath_output = 'result/7495309906/pc_512'
    if not os.path.exists(filepath_data):
        os.makedirs(filepath_data)
    if not os.path.exists(filepath_output):
        os.makedirs(filepath_output)
    fps(id,filepath_data,filepath_output)
    outpath = 'result/7495309906/H5'
    h5(filepath_output, outpath)