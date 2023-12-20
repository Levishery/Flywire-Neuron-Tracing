import os
import numpy as np
import warnings
from plyfile import PlyData
import pickle
import re

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


class FlyTracingDataLoader(Dataset):
    def __init__(self, root, use_features=True, split='train', process_data=False, use_image_feature=None, filter_data=None, test_path=None, daiyi_datapath=False, npoints=1024, return_name=False, block_name=None):
        self.root = root
        self.npoints = npoints
        self.return_name = return_name
        self.process_data = process_data
        self.use_features = use_features
        self.use_image_feature = use_image_feature
        self.filter_data = filter_data
        self.block_name = block_name
        self.feature_dim = 20 if self.use_image_feature else 4
        self.daiyi_datapath = daiyi_datapath
        if self.filter_data:
            self.filter_file = open(filter_data, 'w+')
            self.filter_file = []

        self.catfile = os.path.join(self.root, 'Flytracing_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'FlyTracing_train.txt'))]
        if test_path is None:
            test_path = os.path.join(self.root, 'FlyTracing_test.txt')
        shape_ids['test'] = [line.rstrip() for line in open(test_path)]
        shape_ids['train'] = list(filter(lambda x: 'connector_18_10_121' not in x, shape_ids['train']))
        shape_ids['test'] = list(filter(lambda x: 'connector_18_10_121' not in x, shape_ids['test']))

        assert (split == 'train' or split == 'test')
        shape_names = [x.split('/')[0] for x in shape_ids[split]]
        if not self.daiyi_datapath:
            self.datapath = [
                (shape_names[i], os.path.join(self.root, shape_ids[split][i])) for i in range(len(shape_ids[split]))]
        else:
            self.datapath = [(shape_names[i], os.path.join(self.root.replace('liusl', 'daiyi.zhu'), shape_ids[split][i])) for i
                             in range(len(shape_ids[split]))]
        if self.use_image_feature:
            if 'biological' in self.cat:
                self.image_feature_path = [re.sub(r'evaluate_fafb_dust1200_dis500_rad0.3216_[\d]+', self.block_name, x[1]).replace('biological', self.use_image_feature).replace('.ply', '.h5') for x in self.datapath]
            else:
                self.image_feature_path = [x[1].replace('neg/', self.use_image_feature).replace('pos/', self.use_image_feature).replace('.ply', '.h5') for x in self.datapath]

        print('The size of %s data is %d' % (split, len(self.datapath)))

        # if self.uniform:
        #     self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        # else:
        #     self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # if self.process_data:
        #     if not os.path.exists(self.save_path):
        #         print('Processing data %s (only running in the first time)...' % self.save_path)
        #         self.list_of_points = [None] * len(self.datapath)
        #         self.list_of_labels = [None] * len(self.datapath)
        #
        #         for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
        #             fn = self.datapath[index]
        #             cls = self.classes[self.datapath[index][0]]
        #             cls = np.array([cls]).astype(np.int32)
        #             point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        #
        #             if self.uniform:
        #                 point_set = farthest_point_sample(point_set, self.npoints)
        #             else:
        #                 point_set = point_set[0:self.npoints, :]
        #
        #             self.list_of_points[index] = point_set
        #             self.list_of_labels[index] = cls
        #
        #         with open(self.save_path, 'wb') as f:
        #             pickle.dump([self.list_of_points, self.list_of_labels], f)
        #     else:
        #         print('Load processed data from %s...' % self.save_path)
        #         with open(self.save_path, 'rb') as f:
        #             self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
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


if __name__ == '__main__':
    import torch
    p = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/FlyTracing_train_1.txt'
    data = FlyTracingDataLoader('/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048', split='train', use_image_feature='extract_pc_Unet/', npoints=2048, filter_data=p)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True, num_workers=8)
    log = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_log.txt'
    f_test = open(log, 'w+')
    for batch_id, (points, target) in tqdm(enumerate(DataLoader, 0), total=len(DataLoader),
                                           smoothing=0.9):
        sample = 0
    # data.filter_file.close()
    f_test = open('/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/FlyTracing_train_1.txt', 'w+')
    import pickle
    pickle.dump(data.filter_file, open('/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/FlyTracing_train_1.pickle', 'wb'))
    for string in data.filter_file:
        f_test.writelines(string[6:])
    f_test.close()
        # f_test.writelines(points.shape + '\n')
        # print(point.shape)
        # print(label.shape)
