import os
import random
from tqdm import tqdm
import torch
import pandas as pd
import time

def work(block):
    pc_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/'
    pc_prefix = 'biological'
    feature_prefix = 'biological_feature/extract_biological_pc_Unet'
    block_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/block_with_index'
    file_path_root = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/block_txt'
    block_name = block.split('.')[0]

    if not os.path.exists(os.path.join(file_path_root, block_name+'.txt')):
        t = time.time() - os.path.getmtime(os.path.join(pc_path+feature_prefix, block_name))
        if t > 60:
            samples_record = []
            df = pd.read_csv(os.path.join(block_path, block), header=None)
            N = len(df)
            for ii in tqdm(range(0, N)):
                segid1 = df.iloc[ii, 0]
                segid2 = df.iloc[ii, 1]
                index = df.iloc[ii, 3]
                name = str(segid1) + '_' + str(segid2) + '.ply'
                pc_ply_path = os.path.join(os.path.join(pc_prefix, 'evaluate_fafb_dust1200_dis500_rad0.3216_' + str(index)), name)
                if os.path.exists(os.path.join(pc_path, pc_ply_path)) and os.path.exists(os.path.join(pc_path+feature_prefix, block_name + '/' +name.replace('.ply', '.h5'))):
                    samples_record.append(pc_ply_path)
            if len(samples_record)>0:
                f_test = open(os.path.join(file_path_root, block_name+'.txt'), 'w+')
                for i in range(len(samples_record)):
                    f_test.writelines(samples_record[i] + '\n')
                f_test.close()

if __name__ == '__main__':
    block_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/block_with_index'
    blocks = os.listdir(block_path)
    torch.multiprocessing.set_start_method('spawn')

    num_worker = 8
    num_worker_real = min(num_worker, os.cpu_count())
    if num_worker_real > 0:
        with torch.multiprocessing.Pool(num_worker_real) as pool:
            for res in tqdm(
                    pool.imap_unordered(work, blocks),
                    total=len(blocks),
            ):
                continue
