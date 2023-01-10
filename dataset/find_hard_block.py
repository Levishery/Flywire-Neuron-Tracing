import pandas as pd
import os
import shutil

csv_path = '/braindat/lab/liusl/flywire/block_data/v2/morph_performence.csv'
from_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_4000'
target_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_hard'
df = pd.read_csv(csv_path, index_col=0, header=None)

for i in range(len(df)):
    recall = df.iloc[i].values[1]
    name = df.iloc[i].name
    if recall < 0.72:
        shutil.copy(os.path.join(from_path, name), os.path.join(target_path, name))
