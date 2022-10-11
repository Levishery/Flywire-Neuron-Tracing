import pandas as pd
import os

block_connector = '/braindat/lab/liusl/flywire/block_data/30_percent'
csv_list = os.listdir(block_connector)
stat_dict = {}
for f in csv_list:
    f_name = os.path.join(block_connector, f)
    df = pd.read_csv(f_name, index_col=0)
    stat_dict[f] = len(df)
stat_dict = sorted(stat_dict.items(), key = lambda kv:(kv[1], kv[0]))
print(stat_dict)
