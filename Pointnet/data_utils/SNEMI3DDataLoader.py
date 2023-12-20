import os
import numpy as np
import warnings
from plyfile import PlyData
import pickle
import re

import random
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
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
    return point


class SNEMI3DDataLoader(Dataset):
    def __init__(self, root, use_features=True, split='train', process_data=False, use_image_feature=None, filter_data=None, positive_ratio=0.5, npoints=1024, return_name=False, block_name=None):
        self.root = root
        self.npoints = npoints
        self.return_name = return_name
        self.process_data = process_data
        self.use_features = use_features
        self.use_image_feature = use_image_feature
        self.filter_data = filter_data
        self.positive_ratio = positive_ratio
        self.block_name = block_name
        self.split = split
        self.feature_dim = 20 if self.use_image_feature else 4
        if self.filter_data:
            self.filter_file = open(filter_data, 'w+')
            self.filter_file = []

        self.catfile = os.path.join(self.root, 'SNEMI3D_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if split == 'train':
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'SNEMI3D_train_pos.txt'))] + [line.rstrip() for line in open(os.path.join(self.root, 'SNEMI3D_train_neg.txt'))]
            self.num_pos = len([line.rstrip() for line in open(os.path.join(self.root, 'SNEMI3D_train_pos.txt'))])
            self.num_neg = len([line.rstrip() for line in open(os.path.join(self.root, 'SNEMI3D_train_neg.txt'))])
            self.num_total = self.num_pos + self.num_neg
        else:
            test_path = os.path.join(self.root, 'SNEMI3D_test.txt')
            shape_ids['test'] = [line.rstrip() for line in open(test_path)]

        assert (split == 'train' or split == 'test')
        shape_names = [x.split('/')[0] for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_ids[split][i])) for i
                             in range(len(shape_ids[split]))]
        if self.use_image_feature:
            self.image_feature_path = [x[1].replace('neg/', self.use_image_feature).replace('pos/', self.use_image_feature).replace('.ply', '.h5') for x in self.datapath]

        print('The size of %s data is %d' % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.split != 'test' and self.positive_ratio > 0:
            if random.random() > self.positive_ratio:
                index = self.num_pos + int((index/self.num_total)*self.num_neg)
            else:
                index = int((index/self.num_total)*self.num_pos)
        fn = self.datapath[index]
        if self.use_image_feature:
            imn = self.image_feature_path[index]
        cls = self.classes[self.datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        try:
            pc = PlyData.read(fn[1])
            if self.use_image_feature:
                fid = h5py.File(imn, 'r')
                image_feature = np.array(fid['main'])
            x = pc.elements[0].data['x']
            y = pc.elements[0].data['y']
            z = pc.elements[0].data['z']
            ids = pc.elements[0].data['id']
            if self.use_image_feature:
                point_set = np.concatenate([np.transpose(np.asarray([x, y, z, ids])), image_feature], axis=1)
            else:
                point_set = np.transpose(np.asarray([x, y, z, ids]))
            if self.npoints < len(x):
                point_set = farthest_point_sample(point_set, self.npoints)
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            if not self.use_features:
                point_set = point_set[:, 0:3]
            if self.filter_data:
                # self.filter_file.writelines(fn[1].split('fps')[-1]+'\n')
                self.filter_file.append(fn[1].split('fps')[-1]+'\n')
            assert point_set.shape == (self.npoints, self.feature_dim)
        except:
            print('Warning:', fn[1])
            if self.return_name:
                return np.zeros([self.npoints, 20]), -1, fn[1].split('/')[-1]
            else:
                return np.zeros([self.npoints, 20]), -1

        if self.return_name:
            return point_set, label[0], fn[1].split('/')[-1]
        else:
            return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


