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


def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)


def plot():
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Unetft2/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Baseline_debugged/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/image_only_43k/block_result.csv'
    csv_list = pd.read_csv(path, header=None)
    rec4 = csv_list[1]
    acc4 = csv_list[2]
    labels = 'EdgeNetwork', 'Baseline', 'Ours-Embed.[14]', 'Ours-Image', 'Ours-Pairwise-Embed.'
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    ax1.set_title('Recall')
    bp = ax1.boxplot([rec0, rec2, rec3, rec4, rec1], showmeans=True, showfliers=False, labels=labels)
    print(get_box_plot_data(labels, bp))
    # ax1.set_yscale('log')
    ax1.set_ylim(0.77, 1.01)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    # ax1.set_xlim(0.62, 4.52)
    # show the ytick positions, as a reference

    ax2.set_title('Precision')
    bp = ax2.boxplot([acc0, acc2, acc3, acc4, acc1], showmeans=True, showfliers=False, labels=labels)
    print(get_box_plot_data(labels, bp))
    # ax2.set_yscale('log')
    ax2.set_ylim(0.77, 1.01)
    # ax2.set_xlim(0.63, 4.53)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    # ax1.text(1.5, 0.85, str(np.around(np.mean(rec0), decimals=3)), color="green", fontsize=10, ha='center')
    # ax1.text(2.27, 0.893, str(np.around(np.mean(rec2), decimals=3)), color="green", fontsize=10, ha='center')
    # ax1.text(3.27, 0.9, str(np.around(np.mean(rec3), decimals=3)), color="green", fontsize=10, ha='center')
    # ax1.text(4.27, 0.908, str(np.around(np.mean(rec1), decimals=3)), color="green", fontsize=10, ha='center')
    # ax2.text(1.27, 0.943, str(np.around(np.mean(acc0), decimals=3)), color="green", fontsize=10, ha='center')
    # ax2.text(2.27, 0.943, str(np.around(np.mean(acc2), decimals=3)), color="green", fontsize=10, ha='center')
    # ax2.text(3.27, 0.943, str(np.around(np.mean(acc3), decimals=3)), color="green", fontsize=10, ha='center')
    # ax2.text(4.27, 0.943, str(np.around(np.mean(acc1), decimals=3)), color="green", fontsize=10, ha='center')

    plt.savefig('box.pdf')


def set_new_threshold():
    # pred_path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/predictions'
    # csv_path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/thresh.csv'
    pred_path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/predictions'
    csv_path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/thresh.csv'
    preds = os.listdir(pred_path)
    threshes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for thresh in threshes:
        recall_list = []
        accuracy_list = []
        for pred in tqdm(preds):
            csv_list = pd.read_csv(os.path.join(pred_path, pred), header=None)
            gt = np.asarray(csv_list[2])
            prediction = np.asarray(csv_list[4])
            TP = sum(np.logical_and(prediction > thresh, gt == 1))
            FP = sum(np.logical_and(prediction > thresh, gt == 0))
            FN = sum(np.logical_and(prediction < thresh, gt == 1))
            recall = TP / (TP + FN)
            accuracy = TP / (TP + FP)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
        row = pd.DataFrame(
            [{'thresh': thresh, 'recall': np.mean(recall_list), 'accuracy': np.mean(accuracy_list)}])
        row.to_csv(csv_path, mode='a', header=False, index=False)


def plot_thresh():
    # path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/thresh.csv'
    # csv_list = pd.read_csv(path, header=None)
    # rec1 = csv_list[1]
    # acc1 = csv_list[2]
    # path = '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/thresh.csv'
    # csv_list = pd.read_csv(path, header=None)
    # rec2 = csv_list[1]
    # acc2 = csv_list[2]
    # path = '/braindat/lab/liusl/flywire/experiment/test-3k/Baseline_debugged/thresh.csv'
    # csv_list = pd.read_csv(path, header=None)
    # rec0 = csv_list[1]
    # acc0 = csv_list[2]
    # path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/thresh.csv'
    # csv_list = pd.read_csv(path, header=None)
    # rec3 = csv_list[1]
    # acc3 = csv_list[2]
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(6, 3))
    # plt.gcf().subplots_adjust(bottom=0.13)
    # ax1.plot(rec1, acc1, label='ours')
    # ax1.plot(rec3, acc3, label='Embed.[14]')
    # ax1.plot(rec2, acc2, label='Baseline')
    # ax1.plot(rec0, acc0, label='EdgeNetwork')
    # ax1.plot(rec1[9], acc1[9], 'b^')
    # ax1.plot(rec3[9], acc3[9], '^', color="orange")
    # ax1.plot(rec2[9], acc2[9], 'g^')
    # ax1.plot(rec0[9], acc0[9], 'r^')
    # ax1.legend()
    # ax1.set_title('3,000 Test Blocks')
    # ax1.set_xlabel('recall', labelpad=0.4)
    # ax1.set_ylabel('precision')
    # ax1.set_ylim(0.90, 1.01)
    # ax1.set_xlim(0.80, 1.01)
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    # ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    # ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    # fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=1, ncols=4, figsize=(14, 3.3))
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Baseline_debugged/0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/0dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    ax1.set_title('Misalignment', fontsize=14)
    ax1.plot(rec1, acc1, label='ours')
    ax1.plot(rec3, acc3, label='Embed.[14]')
    ax1.plot(rec2, acc2, label='Baseline')
    ax1.plot(rec0, acc0, label='EdgeNetwork')
    ax1.plot(rec1[9], acc1[9], 'b^')
    ax1.plot(rec3[9], acc3[9], '^', color="orange")
    ax1.plot(rec2[9], acc2[9], 'g^')
    ax1.plot(rec0[9], acc0[9], 'r^')
    ax1.legend(fontsize=11)
    ax1.set_ylim(0.84, 1.0)
    ax1.set_xlim(0.60, 1.0)
    # ax1.set_xlabel('recall', labelpad=0.4, fontsize=13)
    ax1.set_ylabel('precision', fontsize=13)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Baseline_debugged/1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/1dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    ax2.set_title('Missing-Section', fontsize=14)
    ax2.plot(rec1, acc1, label='ours')
    ax2.plot(rec3, acc3, label='Embed.[14]')
    ax2.plot(rec2, acc2, label='Baseline')
    ax2.plot(rec0, acc0, label='EdgeNetwork')
    ax2.plot(rec1[9], acc1[9], 'b^')
    ax2.plot(rec3[9], acc3[9], '^', color="orange")
    ax2.plot(rec2[9], acc2[9], 'g^')
    ax2.plot(rec0[9], acc0[9], 'r^')
    # ax2.legend()
    ax2.set_ylim(0.84, 1.0)
    ax2.set_xlim(0.60, 1.0)
    # ax2.set_xlabel('recall', labelpad=0.4, fontsize=13)
    # ax4.set_ylabel('precision', fontsize=12)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    path = '/braindat/lab/liusl/flywire/experiment/test-3k/finetune_final_best/0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Baseline_debugged/0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/0.5dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    ax3.set_title('Mixed', fontsize=14)
    ax3.plot(rec1, acc1, label='ours')
    ax3.plot(rec3, acc3, label='Embed.[14]')
    ax3.plot(rec2, acc2, label='Baseline')
    ax3.plot(rec0, acc0, label='EdgeNetwork')
    ax3.plot(rec1[9], acc1[9], 'b^')
    ax3.plot(rec3[9], acc3[9], '^', color="orange")
    ax3.plot(rec2[9], acc2[9], 'g^')
    ax3.plot(rec0[9], acc0[9], 'r^')
    # ax2.legend()
    ax3.set_ylim(0.84, 1.0)
    ax3.set_xlim(0.60, 1.0)
    ax3.set_xlabel('recall', labelpad=0.4, fontsize=13)
    ax3.set_ylabel('precision', fontsize=12)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Unetft2/2dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec1 = csv_list[1]
    acc1 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec2 = csv_list[1]
    acc2 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/Baseline_debugged/dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec0 = csv_list[1]
    acc0 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/metric/dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec3 = csv_list[1]
    acc3 = csv_list[2]
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/image_only_43k/2dist_thresh.csv'
    csv_list = pd.read_csv(path, header=None)
    rec4 = csv_list[1]
    acc4 = csv_list[2]
    ax4.set_title('Average', fontsize=14)
    ax4.plot(rec1, acc1, label='ours')
    ax4.plot(rec3, acc3, label='Embed.[14]')
    ax4.plot(rec2, acc2, label='Baseline')
    ax4.plot(rec0, acc0, label='EdgeNetwork')
    ax4.plot(rec4, acc4, label='Image')
    ax4.plot(rec1[9], acc1[9], 'b^')
    ax4.plot(rec3[9], acc3[9], '^', color="orange")
    ax4.plot(rec2[9], acc2[9], 'g^')
    ax4.plot(rec0[9], acc0[9], 'r^')
    ax4.plot(rec0[9], acc0[9], '^', color="purple")
    # ax2.legend()
    # ax4.legend(fontsize=11)
    ax4.set_ylim(0.84, 1.0)
    ax4.set_xlim(0.60, 1.0)
    ax4.set_xlabel('recall', labelpad=0.4, fontsize=13)
    # ax4.set_ylabel('precision', fontsize=12)
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax4.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.18)
    plt.subplots_adjust(wspace=0.18, hspace=0.3)

    plt.savefig('thresh.pdf')


def plot_thresh_distort():
    path = '/braindat/lab/liusl/flywire/experiment/test-3k/image_distortion.xlsx'
    # pred_path = ['/braindat/lab/liusl/flywire/experiment/test-3k/finetune1_43k/predictions',
    #              '/braindat/lab/liusl/flywire/experiment/test-3k/metric/predictions',
    #              '/braindat/lab/liusl/flywire/experiment/test-3k/baseline-55200/predictions',
    #              '/braindat/lab/liusl/flywire/experiment/test-3k/Baseline_debugged/predictions']
    pred_path = ['/braindat/lab/liusl/flywire/experiment/test-3k/Unetft2/predictions']
    list = pd.read_excel(path, header=None)
    target_type = [2, 0, 0.5, 1]
    threshes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # threshes = [0.5]
    for method in pred_path:
        for thresh in threshes:
            recall_list = []
            accuracy_list = []
            for block, t in zip(list[0], list[1]):
                if isinstance(block, str):
                    if t in target_type:
                        if os.path.exists(os.path.join(method, block)):
                            csv_list = pd.read_csv(os.path.join(method, block), header=None)
                            gt = np.asarray(csv_list[2])
                            prediction = np.asarray(csv_list[4])
                            TP = sum(np.logical_and(prediction > thresh, gt == 1))
                            FP = sum(np.logical_and(prediction > thresh, gt == 0))
                            FN = sum(np.logical_and(prediction < thresh, gt == 1))
                            recall = TP / (TP + FN)
                            accuracy = TP / (TP + FP)
                            recall_list.append(recall)
                            accuracy_list.append(accuracy)
            row = pd.DataFrame(
                [{'thresh': thresh, 'recall': np.mean(recall_list), 'accuracy': np.mean(accuracy_list)}])
            row.to_csv(method.replace('predictions', str(target_type[0]) + 'dist_thresh.csv'), mode='a', header=False,
                       index=False)


def stat_biological_recall():
    potential_path = '/braindat/lab/liusl/flywire/block_data/neuroglancer/connector_22_7_97-101.csv'
    block_paths = ['/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_97.csv',
                   '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_98.csv',
                   '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_99.csv',
                   '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_100.csv',
                   '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat/connector_22_7_101.csv']

    edges = pd.read_csv(potential_path, header=None)
    total_positives = 0
    hit = 0
    for block_path in block_paths:
        if os.path.exists(block_path):
            samples = pd.read_csv(block_path, header=None)
            positives_indexes = np.where(samples[3] == 1)
            query = list(samples[0][list(positives_indexes[0])])
            pos = list(samples[1][list(positives_indexes[0])])
            for i in range(len(query)):
                total_positives = total_positives + 1
                potentials = list(edges[1][np.where(edges[0] == query[i])[0]])
                if pos[i] in potentials:
                    hit = hit + 1
                    continue
                potentials = list(edges[1][np.where(edges[0] == pos[i])[0]])
                if query[i] in potentials:
                    hit = hit + 1
                    continue
    print('bilogical edge recall: ', hit / total_positives)


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
