import os
import random
import shutil
from tqdm import tqdm
import torch

source_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_local_1024/'
target_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_local_1024_used'
train_name = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_local_1024/FlyTracing_train.txt'
test_name = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_local_1024/FlyTracing_test.txt'
feature_path = 'extract_pc_Unetz'
# source_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/'
# target_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048_used'
# train_name = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/FlyTracing_train.txt'
# test_name = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/FlyTracing_test.txt'

def chunk2(l,chunk_size):
    return [l[x:x + chunk_size] for x in range(0,len(l),chunk_size)]

import os

import os

def replace_third_line(file_path, replacement_text):
    if file_path.endswith('.ply'):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if len(lines) >= 3:
            lines[2] = replacement_text + '\n' # Python列表索引从0开始，所以第三行对应索引为2

        with open(file_path, 'w') as file:
            file.writelines(lines)

def replace(block_path):
    replacement_text = 'element vertex 2048'
    for filename in tqdm(os.listdir(block_path)):
        if filename.endswith('.ply'): # 你可以根据需要修改文件类型的过滤条件
            file_path = os.path.join(block_path, filename)
            replace_third_line(file_path, replacement_text)



def work(paths):
    for data_path in tqdm(paths):
        os.makedirs(os.path.join(target_path, data_path.split('/')[-3], data_path.split('/')[-2]), exist_ok=True)
        shutil.copy(os.path.join(source_path, data_path), os.path.join(target_path, data_path))

if __name__ == '__main__':
    # f = open(train_name, 'r+')
    # data_paths = [line.rstrip() for line in f]
    # paths = chunk2(data_paths, 1000)
    # torch.multiprocessing.set_start_method('spawn')
    #
    # num_worker = 4
    # num_worker_real = min(num_worker, os.cpu_count())
    # if num_worker_real > 0:
    #     with torch.multiprocessing.Pool(num_worker_real) as pool:
    #         for res in tqdm(
    #                 pool.imap_unordered(work, paths),
    #                 total=len(paths),
    #         ):
    #             continue
    #
    # f = open(test_name, 'r+')
    # data_paths = [line.rstrip() for line in f]
    # for data_path in tqdm(data_paths):
    #     os.makedirs(os.path.join(target_path, data_path.split('/')[-3], data_path.split('/')[-2]), exist_ok=True)
    #     shutil.copy(os.path.join(source_path, data_path), os.path.join(target_path, data_path))
    folder_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps_2048/neg' # 替换为实际的文件夹路径
    replacement_text = 'element vertex 2048'
    block_paths = [os.path.join(folder_path, block_name) for block_name in os.listdir(folder_path)]
    torch.multiprocessing.set_start_method('spawn')

    num_worker = 4
    num_worker_real = min(num_worker, os.cpu_count())
    if num_worker_real > 0:
        with torch.multiprocessing.Pool(num_worker_real) as pool:
            for res in tqdm(
                    pool.imap_unordered(replace, block_paths),
                    total=len(block_paths),
            ):
                continue

