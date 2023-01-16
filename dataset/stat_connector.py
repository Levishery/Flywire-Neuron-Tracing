import pandas as pd
import os
from tqdm import tqdm
import shutil
import numpy as np
import random


def stat():
    block_connector = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    csv_list = os.listdir(block_connector)
    stat_dict = {}
    for f in tqdm(csv_list):
        f_name = os.path.join(block_connector, f)
        df = pd.read_csv(f_name, index_col=0)
        stat_dict[f] = len(df)
    df = pd.DataFrame.from_dict(stat_dict, orient='index', columns=['num_connector'])
    df = df.reset_index().rename(columns = {'index':'block'})
    df = df.sort_values(by="num_connector", ascending=False)
    df.to_csv('/braindat/lab/liusl/flywire/block_data/v2/30_percent.csv', index=False)

def stat_neuron_connector():
    target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    connector_files = os.listdir(target_connector_path)
    valid_neuron_num = 0
    connector_num = []
    for connector_f in tqdm(connector_files):
        df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
        if len(df) > 10:
            valid_neuron_num = valid_neuron_num + 1
            connector_num.append(len(df))
    average_connector = sum(connector_num)/valid_neuron_num
    print(average_connector)

def delete_far():
    connector_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_vali/connector_14_7_167.csv'
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_vali_filtered/connector_14_7_167.csv'
    sample_volume_size = [16,128,128]
    connector_list = pd.read_csv(connector_path, header=None)
    idx = 0
    while idx < len(connector_list):
        connector = connector_list.iloc[idx]
        cord_start_offset = np.asarray(connector[3][1:-1].split(), dtype=np.float32)
        cord_pos_offset = np.asarray(connector[4][1:-1].split(), dtype=np.float32)
        cord_start_offset = np.asarray([cord_start_offset[0] / 4, cord_start_offset[1] / 4, cord_start_offset[2]])
        cord_pos_offset = np.asarray([cord_pos_offset[0] / 4, cord_pos_offset[1] / 4, cord_pos_offset[2]])
        if np.any(np.abs(cord_start_offset - cord_pos_offset) > np.asarray(
            [sample_volume_size[1] - 32, sample_volume_size[2] - 32, sample_volume_size[0] - 2],
            dtype=np.float32)):
            connector_list.drop(idx, inplace=True)
        idx = idx + 1
    connector_list.to_csv(target_path, header=False, index=False)


stat_neuron_connector()

# delete_far()
# stat()
# val_num = 200
# source_dir = '/braindat/lab/liusl/flywire/block_data/2_percent'
# df = pd.read_csv('/braindat/lab/liusl/flywire/block_data/2_percent.csv')
# train_dir = '/braindat/lab/liusl/flywire/block_data/train_2_all'
# val_dir = '/braindat/lab/liusl/flywire/block_data/val_2_all'
#
# # total_num = 4000
# total_num = len(df)
# rows = list(range(0, total_num))
# random.shuffle(rows)
# for i in rows[0:200]:
#     file_name = df['block'][i]
#     shutil.copy(os.path.join(source_dir, file_name), val_dir)
#
# for i in rows[200:]:
#     file_name = df['block'][i]
#     shutil.copy(os.path.join(source_dir, file_name), train_dir)
# source_dir = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
# df = pd.read_csv('/braindat/lab/liusl/flywire/block_data/v2/30_percent.csv')
# train_dir = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train'
#
# total_num = 4000
# for i in range(total_num):
#     file_name = df['block'][i]
#     shutil.copy(os.path.join(source_dir, file_name), train_dir)