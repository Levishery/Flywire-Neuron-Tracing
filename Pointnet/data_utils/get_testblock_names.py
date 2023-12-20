import os
import random
from tqdm import tqdm
import torch

def work(block):
    neg_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/neg'
    pos_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/pos'
    feature_path = ['/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/extract_pc_Unet',
                    '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/extract_pc_Metric',
                    '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/extract_pc_Intensity']
    file_path_root = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/block_names'

    if os.path.exists(os.path.join(pos_path, block)) and not os.path.exists(os.path.join(file_path_root, block+'.txt')):
        if os.path.exists(os.path.join(feature_path[0], block)):
            neg_samples = []
            pos_samples = []
            samples = os.listdir(os.path.join(neg_path, block))
            for sample in samples:
                if os.path.exists(os.path.join(feature_path[0], block, sample.replace('.ply', '.h5'))) and os.path.exists(os.path.join(feature_path[1], block, sample.replace('.ply', '.h5'))) and os.path.exists(os.path.join(feature_path[2], block, sample.replace('.ply', '.h5'))):
                    neg_samples.append(os.path.join('neg', block, sample))
            samples = os.listdir(os.path.join(pos_path, block))
            for sample in samples:
                if os.path.exists(os.path.join(feature_path[0], block, sample.replace('.ply', '.h5'))) and os.path.exists(os.path.join(feature_path[1], block, sample.replace('.ply', '.h5'))) and os.path.exists(os.path.join(feature_path[2], block, sample.replace('.ply', '.h5'))):
                    pos_samples.append(os.path.join('pos', block, sample))
            if len(pos_samples)>0 and len(neg_samples)>0:
                f_test = open(os.path.join(file_path_root, block+'.txt'), 'w+')
                for i in range(len(neg_samples)):
                    f_test.writelines(neg_samples[i] + '\n')
                for i in range(len(pos_samples)):
                    f_test.writelines(pos_samples[i] + '\n')
                f_test.close()

if __name__ == '__main__':
    neg_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/neg'
    neg_blocks = os.listdir(neg_path)
    torch.multiprocessing.set_start_method('spawn')

    num_worker = 8
    num_worker_real = min(num_worker, os.cpu_count())
    if num_worker_real > 0:
        with torch.multiprocessing.Pool(num_worker_real) as pool:
            for res in tqdm(
                    pool.imap_unordered(work, neg_blocks),
                    total=len(neg_blocks),
            ):
                continue



