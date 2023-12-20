import os
import random
from tqdm import tqdm

prifixs = ['train', 'test']
for prefix in prifixs:
    neg_path = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/neg'
    pos_path = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/pos'
    feature_path1 = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/connect-embed'
    feature_path2 = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/seg-embed'
    prefix_name_pos = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/SNEMI3D_{prefix}_pos.txt'
    prefix_name_neg = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/SNEMI3D_{prefix}_neg.txt'
    prefix_name = f'/h3cstore_nt/JaneChen/SNEMI3D/pc-{prefix}-2048/SNEMI3D_{prefix}.txt'
    neg_samples = []
    pos_samples = []

    samples = os.listdir(neg_path)
    for sample in samples:
        if os.path.exists(os.path.join(feature_path1, sample.replace('ply', 'h5'))) and os.path.exists(os.path.join(feature_path2, sample.replace('ply', 'h5'))):
            neg_samples.append(os.path.join('neg', sample))
        else:
            print(os.path.join(feature_path1, sample.replace('ply', 'h5')))
    samples = os.listdir(pos_path)
    for sample in samples:
        if os.path.exists(os.path.join(feature_path1, sample.replace('ply', 'h5'))) and os.path.exists(os.path.join(feature_path2, sample.replace('ply', 'h5'))):
            pos_samples.append(os.path.join('pos', sample))
        else:
            print(os.path.join(feature_path1, sample.replace('ply', 'h5')))

    pos_num = len(pos_samples)
    neg_num = len(neg_samples)
    random.shuffle(pos_samples)
    random.shuffle(neg_samples)

    if prefix == 'train':
        f_prefix_pos = open(prefix_name_pos, 'w+')
        for i in range(len(pos_samples)):
            f_prefix_pos.writelines(pos_samples[i]+'\n')

        f_prefix_neg = open(prefix_name_neg, 'w+')
        for i in range(len(neg_samples)):
            f_prefix_neg.writelines(neg_samples[i]+'\n')
    else:
        f_prefix = open(prefix_name, 'w+')
        for i in range(len(pos_samples)):
            f_prefix.writelines(pos_samples[i]+'\n')
        for i in range(len(neg_samples)):
            f_prefix.writelines(neg_samples[i]+'\n')




