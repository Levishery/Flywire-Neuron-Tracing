import os
import random
from tqdm import tqdm

neg_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/neg'
pos_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/pos'
feature_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/extract_pc_Unet'
feature_path_ = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/extract_pc_Metric'
train_name = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/FlyTracing_train_f.txt'
test_name = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps_2048/FlyTracing_test_f.txt'
train_ratio = 0.98
neg_ratio = 7.1/3
neg_blocks = os.listdir(neg_path)
neg_samples = []
pos_samples = []

for block in tqdm(neg_blocks):
    samples = os.listdir(os.path.join(neg_path, block))
    for sample in samples:
        if os.path.exists(os.path.join(feature_path_, block, sample.replace('ply', 'h5'))) and os.path.exists(os.path.join(feature_path, block, sample.replace('ply', 'h5'))):
            neg_samples.append(os.path.join('neg', block, sample))
        else:
            print(os.path.join('neg', block, sample))
    if os.path.exists(os.path.join(pos_path, block)):
        samples = os.listdir(os.path.join(pos_path, block))
        for sample in samples:
            if os.path.exists(os.path.join(feature_path_, block, sample.replace('ply', 'h5'))) and os.path.exists(
                    os.path.join(feature_path, block, sample.replace('ply', 'h5'))):
                pos_samples.append(os.path.join('pos', block, sample))
            else:
                print(os.path.join('pos', block, sample))

pos_num = len(pos_samples)
neg_num = pos_num*neg_ratio
random.shuffle(pos_samples)
random.shuffle(neg_samples)
neg_samples = neg_samples[:int(neg_num)]

f_train = open(train_name, 'w+')
f_test = open(test_name, 'w+')
for i in range(len(pos_samples)):
    if i < len(pos_samples)*train_ratio:
        f_train.writelines(pos_samples[i]+'\n')
    else:
        f_test.writelines(pos_samples[i]+'\n')
for i in range(len(neg_samples)):
    if i < len(neg_samples)*train_ratio:
        f_train.writelines(neg_samples[i]+'\n')
    else:
        f_test.writelines(neg_samples[i]+'\n')





