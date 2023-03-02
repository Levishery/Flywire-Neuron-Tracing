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


# filepath_data = r"E:\Desktop\aw"
# '/braindat/lab/liusl/flywire/block_data/train_30'
filepath_data = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train_1000_reformat'
filenames = os.listdir(filepath_data)
q = 0
n = len(filenames)
print(n)
lx = 80
ly = 80
lz = 32
vol = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True, parallel=True)
pos_path_output = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train/pos'
neg_path_output = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train/neg'
step = 1

for q in tqdm(range(0, n)):
    # file_name = 'connector_17_12_143.csv'
    file_name = filenames[q]
    path = os.path.join(filepath_data, str(file_name))
    df = pd.read_csv(path, header=None)
    # print(df)
    # #print(df.iloc[3,:])
    N = len(df)
    ii = 0
    pos_filepath_output = os.path.join(pos_path_output, str(file_name).split('.')[0])
    neg_filepath_output = os.path.join(neg_path_output, str(file_name).split('.')[0])
    if not os.path.exists(pos_filepath_output):
        os.makedirs(neg_filepath_output)
        os.makedirs(pos_filepath_output)
        for ii in range(0, N):
            label = df.iloc[ii, 3]
            if label == 1:
                filepath_output = pos_filepath_output
            else:
                filepath_output = neg_filepath_output
            segid1 = df.iloc[ii, 0]
            segid2 = df.iloc[ii, 1]
            # if segid1 == 6978865582:
            #     print('@')
            cord = df.iloc[ii, 5][1:-1].split(',')
            # cord = df.iloc[ii, 2][1:-1].split()
            # cord = [int(cord[0]), int(cord[1]), int(cord[2]), int(cord[3])]
            # cord = np.asarray(cord) + np.asarray([0, 8, 64, 64]) + np.asarray([0, 16+28, 156, 156])
            # fafb_cord = xpblock_to_fafb()

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
                fid1.write(str((boundary0[0, 0] + x - lx) * 16) + "\t" + str((boundary0[0, 1] + y - ly) * 16))
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
                        fid1.write(str((boundary0[n, 0] - lx + x) * 16) + "\t" + str((boundary0[n, 1] - ly + y) * 16))
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
                fid1.write(str((boundary1[0, 0] - lx + x) * 16) + "\t" + str((boundary1[0, 1] - ly + y) * 16))
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
                        fid1.write(str((boundary1[n, 0] - lx + x) * 16) + "\t" + str((boundary1[n, 1] - ly + y) * 16))
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
    else:
        print('%s exists.'%neg_filepath_output)
