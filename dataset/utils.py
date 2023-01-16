import numpy as np
import math
import pandas as pd
import os
import shutil
from matplotlib import pyplot as plt

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
    z_coo:xp集群点所在块内的z轴坐标
    y_coo:xp集群点所在块内的y轴坐标
    x_coo:xp集群点所在块内的x轴坐标

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
        return x_block, y_block, z_block, x_pixel,y_pixel,z_pixel
    else:
        return x_block, y_block, z_block

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
    thresh = recall_list[int(len(df)*0.15)]
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