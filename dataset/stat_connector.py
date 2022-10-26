import pandas as pd
import os
from tqdm import tqdm
import shutil
import random


def stat():
    block_connector = '/braindat/lab/liusl/flywire/block_data/30_percent'
    csv_list = os.listdir(block_connector)
    stat_dict = {}
    for f in tqdm(csv_list):
        f_name = os.path.join(block_connector, f)
        df = pd.read_csv(f_name, index_col=0)
        stat_dict[f] = len(df)
    df = pd.DataFrame.from_dict(stat_dict, orient='index', columns=['num_connector'])
    df = df.reset_index().rename(columns = {'index':'block'})
    df = df.sort_values(by="num_connector", ascending=False)
    df.to_csv('/braindat/lab/liusl/flywire/block_data/30_percent.csv', index=False)

val_num = 200
source_dir = '/braindat/lab/liusl/flywire/block_data/2_percent'
df = pd.read_csv('/braindat/lab/liusl/flywire/block_data/2_percent.csv')
train_dir = '/braindat/lab/liusl/flywire/block_data/train_2_all'
val_dir = '/braindat/lab/liusl/flywire/block_data/val_2_all'

# total_num = 4000
total_num = len(df)
rows = list(range(0, total_num))
random.shuffle(rows)
for i in rows[0:200]:
    file_name = df['block'][i]
    shutil.copy(os.path.join(source_dir, file_name), val_dir)

for i in rows[200:]:
    file_name = df['block'][i]
    shutil.copy(os.path.join(source_dir, file_name), train_dir)
