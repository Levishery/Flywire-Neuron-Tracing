import math
import os

import cv2
import cc3d
import numpy as np
import argparse
# /braindat/lab/liusl/flywire/block_data/30_percent
import pandas as pd
import shutil
from matplotlib import pyplot as plt
from utils import readh5
from tqdm import tqdm
from plyfile import PlyData
from utils import writeh5
from connectomics.data.utils import select_points, get_crop_index, readh5


if __name__ == '__main__':
    prefixes = ['train', 'test']
    for prefix in prefixes:
        file_name = f'/h3cstore_nt/JaneChen/SNEMI3D/edges-0nm-8x64x64/{prefix}ing/SNEMI3D-{prefix}-segmentation-wos-reduced-nodes-0-17x129x129-SNEMI3D.csv'
        sampled_pos_path = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/pos'
        sampled_neg_path = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/neg'
        pc_result_path = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/connect-embed'
        image_feature_patch_path = f'/h3cstore_nt/JaneChen/SNEMI3D/image_feature_patch/connect-embed/{prefix}'

        pc_patch_size = np.asarray([321, 321, 65])
        patch_size = np.asarray([129, 129, 17])
        half_patch_size = np.asarray([3, 3, 1])
        patch_cord = (pc_patch_size - np.asarray([2, 2, 1]) * patch_size)/2
        df = pd.read_csv(file_name)
        N = len(df)
        ii = 0
        for ii in tqdm(range(0, N)):
            label = df.iloc[ii, 5]
            if label == 1:
                pc_path = sampled_pos_path
            elif label == 0:
                pc_path = sampled_neg_path
            else:
                continue
            segid1 = df.iloc[ii, 1]
            segid2 = df.iloc[ii, 2]
            name = str(segid1) +'_' + str(segid2) + '.ply'
            filename_pcd = os.path.join(pc_path, name)
            pc = PlyData.read(filename_pcd)
            patch_file = os.path.join(image_feature_patch_path, name.replace('.ply', '.h5'))
            if os.path.exists(patch_file):
                data = readh5(patch_file)
            else:
                name_ = str(segid2) + '_' + str(segid1) + '.h5'
                patch_file = os.path.join(image_feature_patch_path, name_)
                data = readh5(patch_file)
            embedding_patch = data[:16, :, :, :]
            mask_patch = data[-1, :, :, :]
            mask_0 = mask_patch == segid1
            mask_1 = mask_patch == segid2
            mask = np.concatenate((np.expand_dims(mask_0, 0), np.expand_dims(mask_1, 0)), axis=0)

            x = pc.elements[0].data['x']
            y = pc.elements[0].data['y']
            z = pc.elements[0].data['z']
            ids = pc.elements[0].data['id']
            cords = np.transpose(
                np.asarray([x, y, z]) / np.expand_dims(np.asarray([6, 6, 30]), axis=1),
                [1, 0])

            bbox = [list(patch_cord), list(patch_cord + patch_size * [2, 2, 1])]
            in_box_indexes = np.where(select_points(bbox, cords))[0]
            embeddings = np.zeros([len(x), 16])
            for in_box_index in in_box_indexes:
                cord_in_patch = np.round((cords[in_box_index] - patch_cord) / [2, 2, 1]).astype(
                    np.int32)
                [x_bound, y_bound, z_bound] = get_crop_index(cord_in_patch, half_patch_size,
                                                             patch_size)
                # mask the local embedding with interested segmentation mask
                local_mask = mask[ids[in_box_index], z_bound[0]:z_bound[1],
                             y_bound[0]:y_bound[1], x_bound[0]:x_bound[1]] == 1

                # if not local_mask.any():
                #     print('debug')

                local_embed = embedding_patch[:, z_bound[0]:z_bound[1], y_bound[0]:y_bound[1],
                              x_bound[0]:x_bound[1]]
                mean_embed = np.mean(local_embed[:, local_mask], axis=1)
                embeddings[in_box_index, :] = mean_embed
            writeh5(os.path.join(pc_result_path, name.replace('.ply', '.h5')), embeddings)

