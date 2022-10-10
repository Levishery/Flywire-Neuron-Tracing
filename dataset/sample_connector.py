import pandas as pd
import os
import navis
import numpy as np
from tqdm import tqdm
import math

z_block = 272
y_block = 22
x_block = 41


def fafb_to_block(x, y, z):
    '''
    (x,y,z):fafb坐标
    (x_block,y_block,z_block):block块号
    (x_pixel,y_pixel,z_pixel):块内像素号,其中z为帧序号,为了使得z属于中间部分,强制z属于[29,54)
    文件名：z/y/z-xxx-y-xx-x-xx
    '''
    x_block_float = (x + 17631) / 1736 / 4
    y_block_float = (y + 19211) / 1736 / 4
    z_block_float = (z - 15) / 26
    x_block = math.floor(x_block_float)
    y_block = math.floor(y_block_float)
    z_block = math.floor(z_block_float)
    x_pixel = (x_block_float - x_block) * 1736
    y_pixel = (y_block_float - y_block) * 1736
    z_pixel = (z - 15) - z_block * 26
    while z_pixel < 28:
        z_block = z_block - 1
        z_pixel = z_pixel + 26
    return x_block, y_block, z_block

def define_block_list():
    block_list = [[[]] * x_block for i in range(y_block)]
    block_list = [block_list for i in range(z_block)]
    return block_list


target_tree_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data'
target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
visualization_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/visualization'
connector_files = os.listdir(target_connector_path)
block_connector = '/braindat/lab/liusl/flywire/block_data/30_percent'
thresh = 10

for connector_f in tqdm(connector_files):
    # block_list = define_block_list()
    visualization_f = connector_f.replace('_connector.csv', '_save.csv')
    neuron_id = visualization_f[:-9]
    tree_f = connector_f.replace('_connector.csv', '.json')
    tree = navis.read_json(os.path.join(target_tree_path, tree_f))[0]
    max_strahler_index = tree.nodes.strahler_index.max()
    df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
    df_weight = pd.read_csv(os.path.join(visualization_path, visualization_f), index_col=0)

    for i in df.index:
        score1 = 1 / ((1 / min(df_weight[str(int(df['node1_segid'][i]))][0], 40)) * (
                    1 / min(df_weight[str(int(df['node0_segid'][i]))][0], 40)))
        score2 = (1 / (max_strahler_index - int(df['Strahler order'][i].split('\n')[0].split(' ')[-1]) + 3))
        score = score2 * score1
        if score > thresh:
            cord0 = np.array(df['node0_cord'][i][1:-1].split(', ')).astype(float)
            cord1 = np.array(df['node1_cord'][i][1:-1].split(', ')).astype(float)
            cord = (cord0 + cord1) / 2
            x_b, y_b, z_b = fafb_to_block(cord[0], cord[1], cord[2])
            file_name = 'connector_' + str(x_b) + '_' + str(y_b) + '_' + str(z_b) + '.csv'
            row = pd.DataFrame([{'node0_segid': int(df['node0_segid'][i]), 'node1_segid': int(df['node1_segid'][i]),
                                'cord': cord, 'score': score, 'neuron_id': neuron_id}])
            row.to_csv(os.path.join(block_connector, file_name), mode='a', header=True, index=False)

            # block_list[z_b][y_b][x_b].append([int(df['node0_segid'][i]), int(df['node1_segid'][i]),
            #                                                    cord, score, neuron_id])
        # for i in range(x_block):
        #     for j in range(y_block):
        #         for k in range(z_block):
        #             if len(block_list[k][j][i]) > 0:
        #                 file_name = 'connector_' + str(i) + '_' + str(j) + '_' + str(k) + '.csv'
        #                 df = pd.DataFrame(block_list[k][j][i], columns=['node0_segid', 'node1_segid', 'cord',
        #                                                                 'score', 'neuron_id'])
        #                 df.to_csv(file_name, mode='a', header=True, index=False)
