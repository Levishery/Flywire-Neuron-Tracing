import numpy as np
import math
import pandas as pd
import os
import shutil
from tqdm import tqdm
import random
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
import navis

z_block = 272
y_block = 22
x_block = 41


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def xpblock_to_fafb(z_block, y_block, x_block, z_coo=0, y_coo=0, x_coo=0):
    '''
    函数功能:xp集群上的点到fafb原始坐标的转换
    输入:xp集群点属于的块儿和块内坐标,每个块的大小为1736*1736*26(x,y,z顺序)
    输出:fafb点坐标
    z_block:xp集群点做所在块儿的z轴序号
    y_block:xp集群点做所在块儿的y轴序号
    x_clock:xp集群点做所在块儿的x轴序号
    z_coo:xp集群点所在块内的z轴坐标 (29-54)
    y_coo:xp集群点所在块内的y轴坐标 (0-1735)
    x_coo:xp集群点所在块内的x轴坐标 (0-1735)

    '''

    # 第二个：原来的,z准确的,x,y误差超过5,block大小为 26*1736*1736
    # x_fafb = 1736 * 4 * x_block - 17830 + 4 * x_coo
    x_fafb = 1736 * 4 * x_block - 17631 + 4 * x_coo
    # y_fafb = 1736 * 4 * y_block - 19419 + 4 * y_coo
    y_fafb = 1736 * 4 * y_block - 19211 + 4 * y_coo
    z_fafb = 26 * z_block + 15 + z_coo
    # z_fafb = 26 * z_block + 30 + z_coo
    return (x_fafb, y_fafb, z_fafb)


def fafb_to_block(x, y, z, return_pixel=False):
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
    if return_pixel:
        return x_block, y_block, z_block, x_pixel, y_pixel, z_pixel
    else:
        return x_block, y_block, z_block


def define_block_list():
    block_list = [[[]] * x_block for i in range(y_block)]
    block_list = [block_list for i in range(z_block)]
    return block_list


def split_train_test_block():
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
    source_dir = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    df = pd.read_csv('/braindat/lab/liusl/flywire/block_data/v2/30_percent.csv')
    train_dir = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train'

    total_num = 4000
    for i in range(total_num):
        file_name = df['block'][i]
        shutil.copy(os.path.join(source_dir, file_name), train_dir)


def sample_by_block():
    target_tree_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data'
    target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    visualization_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/visualization'
    connector_files = os.listdir(target_connector_path)
    block_connector = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    thresh = 10

    for connector_f in tqdm(connector_files):
        # block_list = define_block_list()
        visualization_f = connector_f.replace('_connector.csv', '_save.csv')
        neuron_id = visualization_f[:-9]
        df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
        df_weight = pd.read_csv(os.path.join(visualization_path, visualization_f), index_col=0)
        if len(df) > 10:
            tree_f = connector_f.replace('_connector.csv', '.json')
            tree = navis.read_json(os.path.join(target_tree_path, tree_f))[0]
            max_strahler_index = tree.nodes.strahler_index.max()

            for i in df.index:
                weight1 = df_weight[str(int(df['node1_segid'][i]))][0]
                weight0 = df_weight[str(int(df['node0_segid'][i]))][0]
                score1 = 1 / ((1 / min(weight1, 40)) * (1 / min(weight0, 40)))
                score2 = (1 / (max_strahler_index - int(df['Strahler order'][i].split('\n')[0].split(' ')[-1]) + 3))
                score = score2 * score1
                if score > thresh:
                    cord0 = np.array(df['node0_cord'][i][1:-1].split(', ')).astype(float)
                    cord1 = np.array(df['node1_cord'][i][1:-1].split(', ')).astype(float)
                    cord = (cord0 + cord1) / 2
                    x_b, y_b, z_b = fafb_to_block(cord[0], cord[1], cord[2])
                    file_name = 'connector_' + str(x_b) + '_' + str(y_b) + '_' + str(z_b) + '.csv'
                    # node 0 is seg_start, which is a larger segment.
                    if weight0 > weight1:
                        row = pd.DataFrame(
                            [{'node0_segid': int(df['node0_segid'][i]), 'node1_segid': int(df['node1_segid'][i]),
                              'cord': cord, 'cord0': cord0, 'cord1': cord1, 'score': score, 'neuron_id': neuron_id}])
                    else:
                        row = pd.DataFrame(
                            [{'node0_segid': int(df['node1_segid'][i]), 'node1_segid': int(df['node0_segid'][i]),
                              'cord': cord, 'cord0': cord1, 'cord1': cord0, 'score': score, 'neuron_id': neuron_id}])
                    row.to_csv(os.path.join(block_connector, file_name), mode='a', header=False, index=False)
        else:
            print('invalid neuron', neuron_id)
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


def sample_by_neuron():
    target_tree_path = '/braindat/lab/liusl/flywire/test-skel/tree_data'
    target_connector_path = '/braindat/lab/liusl/flywire/test-skel/connector_data'
    visualization_path = '/braindat/lab/liusl/flywire/test-skel/visualization'
    connector_files = os.listdir(target_connector_path)
    block_connector = '/braindat/lab/liusl/flywire/test-skel/30_percent'
    thresh = 10

    for connector_f in tqdm(connector_files):
        # block_list = define_block_list()
        visualization_f = connector_f.replace('_connector.csv', '_save.csv')
        neuron_id = visualization_f[:-9]
        print('processing neuron', neuron_id)
        df = pd.read_csv(os.path.join(target_connector_path, connector_f), index_col=0)
        df_weight = pd.read_csv(os.path.join(visualization_path, visualization_f), index_col=0)
        if len(df) > 10:
            tree_f = connector_f.replace('_connector.csv', '.json')
            tree = navis.read_json(os.path.join(target_tree_path, tree_f))[0]
            max_strahler_index = tree.nodes.strahler_index.max()

            for i in df.index:
                weight1 = df_weight[str(int(df['node1_segid'][i]))][0]
                weight0 = df_weight[str(int(df['node0_segid'][i]))][0]
                score1 = 1 / ((1 / min(weight1, 40)) * (1 / min(weight0, 40)))
                score2 = (1 / (max_strahler_index - df['Strahler order'][i] + 3))
                score = score2 * score1
                if score > thresh:
                    cord0 = np.array(df['node0_cord'][i][1:-1].split(', ')).astype(float)
                    cord1 = np.array(df['node1_cord'][i][1:-1].split(', ')).astype(float)
                    cord = (cord0 + cord1) / 2
                    file_name = str(neuron_id) + '.csv'
                    # node 0 is seg_start, which is a larger segment.
                    if weight0 > weight1:
                        row = pd.DataFrame(
                            [{'node0_segid': int(df['node0_segid'][i]), 'node1_segid': int(df['node1_segid'][i]),
                              'cord': cord, 'cord0': cord0, 'cord1': cord1, 'score': score, 'neuron_id': neuron_id}])
                    else:
                        row = pd.DataFrame(
                            [{'node0_segid': int(df['node1_segid'][i]), 'node1_segid': int(df['node0_segid'][i]),
                              'cord': cord, 'cord0': cord1, 'cord1': cord0, 'score': score, 'neuron_id': neuron_id}])
                    row.to_csv(os.path.join(block_connector, file_name), mode='a', header=False, index=False)
        else:
            print('invalid neuron', neuron_id)
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


def find_hard_blocks():
    csv_path = '/braindat/lab/liusl/flywire/block_data/v2/morph_performence.csv'
    from_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_4000'
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_hard'
    df = pd.read_csv(csv_path, index_col=0, header=None)

    for i in range(len(df)):
        recall = df.iloc[i].values[1]
        name = df.iloc[i].name
        if recall < 0.72:
            shutil.copy(os.path.join(from_path, name), os.path.join(target_path, name))


def select_hard():
    block_connector = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_4000'
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train_1000'
    target_hard_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_train_1000_hard'
    target_test_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000'
    result_name = '/braindat/lab/liusl/flywire/block_data/v2/morph_performence_1k.csv'
    df = pd.read_csv(result_name, index_col=0, header=None)
    recall_list = []
    for i in range(len(df)):
        f_name = df.iloc[i].name
        # shutil.copy(os.path.join(block_connector, f_name), os.path.join(target_path, f_name))
        recall = df.iloc[i].values[1]
        recall_list.append(recall)
    recall_list.sort()
    # plt.hist(recall_list)
    # plt.show()
    thresh = recall_list[int(len(df) * 0.15)]
    name_list = []
    for i in range(len(df)):
        f_name = df.iloc[i].name
        name_list.append(f_name)
        recall = df.iloc[i].values[1]
        # if recall < thresh:
        #     shutil.copy(os.path.join(block_connector, f_name), os.path.join(target_hard_path, f_name))
    connector_list = os.listdir(block_connector)
    for f in connector_list:
        if f in name_list:
            continue
        shutil.copy(os.path.join(block_connector, f), os.path.join(target_test_path, f))


def get_neuron_list():
    proofread_status = '/braindat/lab/liusl/flywire/proof_stat_df.csv'
    used_neurons_file = '/braindat/lab/liusl/flywire/used_neurons.csv'
    target_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    connector_files = os.listdir(target_connector_path)
    connector_files = [int(x.split('_')[0]) for x in connector_files]
    proof_list = pd.read_csv(proofread_status)
    used_neurons = pd.DataFrame(columns=proof_list.columns)
    frames = []
    record = []
    for i in tqdm(range(len(proof_list))):
        neuron_id = int(proof_list['pt_root_id'][i])
        if neuron_id in connector_files and neuron_id not in record:
            row = proof_list.iloc[i]
            frames.append(row)
            record.append(neuron_id)
    used_neurons = pd.concat(frames, axis=1).transpose()
    used_neurons.to_csv(used_neurons_file)


def stat():
    block_connector = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    csv_list = os.listdir(block_connector)
    stat_dict = {}
    for f in tqdm(csv_list):
        f_name = os.path.join(block_connector, f)
        df = pd.read_csv(f_name, index_col=0)
        stat_dict[f] = len(df)
    df = pd.DataFrame.from_dict(stat_dict, orient='index', columns=['num_connector'])
    df = df.reset_index().rename(columns={'index': 'block'})
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
    average_connector = sum(connector_num) / valid_neuron_num
    print(average_connector)

def plot():
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/EdgeNEtwork/block_result_02.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    labels = 'EdgeNetwork', 'Baseline', 'Ours'
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    ax1.set_title('Recall')
    ax1.boxplot([rec0, rec2, rec1], showmeans=True, showfliers=False, labels=labels, )
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    # show the ytick positions, as a reference

    ax2.set_title('Precision')
    ax2.boxplot([acc0, acc2, acc1], showmeans=True, showfliers=False, labels=labels)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    ax1.text(1.4, 0.87, str(np.around(np.mean(rec0), decimals=3)), color="green", fontsize=10, ha='center')
    ax1.text(2.27, 0.893, str(np.around(np.mean(rec2), decimals=3)), color="green", fontsize=10, ha='center')
    ax1.text(3.27, 0.908, str(np.around(np.mean(rec1), decimals=3)), color="green", fontsize=10, ha='center')

    ax2.text(1.4, 0.923, str(np.around(np.mean(acc0), decimals=3))+'0', color="green", fontsize=10, ha='center')
    ax2.text(2.27, 0.947, str(np.around(np.mean(acc2), decimals=3)), color="green", fontsize=10, ha='center')
    ax2.text(3.27, 0.949, str(np.around(np.mean(acc1), decimals=3)), color="green", fontsize=10, ha='center')

    plt.savefig('box.pdf')


def set_new_threshold():
    pred_path = '/braindat/lab/liusl/flywire/experiment/test-3k/EdgeNEtwork/predictions'
    csv_path = '/braindat/lab/liusl/flywire/experiment/test-3k/EdgeNEtwork/block_result_02.csv'
    preds = os.listdir(pred_path)
    for pred in tqdm(preds):
        csv_list = pd.read_csv(os.path.join(pred_path, pred), header=None)
        gt = np.asarray(csv_list[2])
        prediction = np.asarray(csv_list[4])
        TP = sum(np.logical_and(prediction > 0.2, gt == 1))
        FP = sum(np.logical_and(prediction > 0.2, gt == 0))
        FN = sum(np.logical_and(prediction < 0.2, gt == 1))
        recall = TP / (TP + FN)
        accuracy = TP / (TP + FP)
        row = pd.DataFrame(
            [{'block_name': pred, 'recall': recall, 'accuracy': accuracy}])
        row.to_csv(csv_path, mode='a', header=False, index=False)


def delete_far():
    connector_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_vali/connector_14_7_167.csv'
    target_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_vali_filtered/connector_14_7_167.csv'
    sample_volume_size = [16, 128, 128]
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
