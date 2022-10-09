import pandas as pd
import os
import navis
import numpy as np
from tqdm import tqdm

target_tree_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data'
target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
visualization_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/visualization'
connector_files = os.listdir(target_connector_path)

connector_table = {'node0_segid': [], 'node1_segid': [], 'cord': [], 'score': []}
connector_table = pd.DataFrame(connector_table)
for connector_f in tqdm(connector_files[0:50]):
    visualization_f = connector_f.replace('_connector.csv', '_save.csv')
    neuron_id = visualization_f[:-9]
    tree_f = connector_f.replace('_connector.csv', '.json')
    tree = navis.read_json(os.path.join(target_tree_path, tree_f))[0]
    max_strahler_index = tree.nodes.strahler_index.max()
    df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
    df_weight = pd.read_csv(os.path.join(visualization_path, visualization_f), index_col=0)
    for i in df.index:
        score1 = 1/((1/min(df_weight[str(int(df['node1_segid'][i]))][0], 40)) * (1/min(df_weight[str(int(df['node0_segid'][i]))][0], 40)))
        score2 = (1 /(max_strahler_index - int(df['Strahler order'][i].split('\n')[0].split(' ')[-1])+3))
        score = score2*score1
        cord0 = np.array(df['node0_cord'][i][1:-1].split(', ')).astype(float)
        cord1 = np.array(df['node1_cord'][i][1:-1].split(', ')).astype(float)
        connector_table.loc[len(connector_table.index)] = [int(df['node0_segid'][i]), int(df['node1_segid'][i]),
                                                           (cord0 + cord1)/2, score]
connector_table = connector_table.sort_values(by='score')
connector_table.to_csv('score.csv')



