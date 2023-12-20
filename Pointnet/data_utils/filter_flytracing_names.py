import os
import random

test_name = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps/FlyTracing_test.txt'
target_test_name = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_fps/FlyTracing_test_1:1.txt'
neg_ratio = 1
neg_samples = []
pos_samples = []

samples = [line.rstrip() for line in open(test_name)]
for sample in samples:
    if sample.split('/')[0] == 'neg':
        neg_samples.append(sample)
    else:
        pos_samples.append(sample)

pos_num = len(pos_samples)
neg_num = pos_num*neg_ratio
random.shuffle(pos_samples)
random.shuffle(neg_samples)
neg_samples = neg_samples[:int(neg_num)]

f_test = open(target_test_name, 'w+')
for i in range(len(pos_samples)):
        f_test.writelines(pos_samples[i]+'\n')
for i in range(len(neg_samples)):
        f_test.writelines(neg_samples[i]+'\n')





