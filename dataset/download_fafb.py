import os.path

import cloudvolume
import pandas as pd
import math
from glob import glob
import numpy as np
import signal
import shutil
from PIL import Image

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Get GT voxel blocks for one neuron.")
    parser.add_argument('--block_id', type=str, default='None',
                        help='block_id (connector_x_y_z) to be download')
    args = parser.parse_args()
    return args


def xpblock_to_fafb(z_block, y_block, x_block, z_coo, y_coo, x_coo):
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

def zfill_all(path):
    list = glob(path+'/*.png')
    for l in list:
        print('moving', l)
        print('to ', l.split('\\')[-1].split('.')[0].zfill(4) + '.png')
        shutil.move(l, os.path.join(path, l.split('\\')[-1].split('.')[0].zfill(4) + '.png'))

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
    return x_block, y_block, z_block, x_pixel,y_pixel,z_pixel

class TimeOutException(Exception):
    pass


def handle(signum, frame):
    raise TimeOutException("运行超时！")


def set_timeout(timeout, callback):
    def wrapper(func):
        def inner(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(timeout)  # 开启闹钟信号
                rs = func(*args, **kwargs)
                signal.alarm(0)  # 关闭闹钟信号
                return rs
            except TimeOutException as e:
                callback()

        return inner

    return wrapper


def process_time():
    print("超时了")


@set_timeout(300, process_time)
def get_volume(start_x,end_x, start_y,end_y, start_z,end_z, source):
    return source[start_x:end_x, start_y:end_y, start_z:end_z]


def download():
    args = get_args()
    block_name = args.block_id
    fafb_v14 = cloudvolume.CloudVolume('https://storage.googleapis.com/neuroglancer-fafb-data/fafb_v14/fafb_v14_orig/',
                                       mip=2, progress=True)
    if block_name != 'None':
        file_name = block_name
        folder_name = file_name.split('.')[0]
        if not os.path.exists(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14', folder_name)):
            [block_x, block_y, block_z] = file_name.split('_')[1:]
            block_x = int(block_x)
            block_y = int(block_y)
            block_z = int(block_z.split('.')[0])
            (start_x, start_y, start_z) = xpblock_to_fafb(block_z, block_y, block_x, 29, 0, 0)
            (end_x, end_y, end_z) = xpblock_to_fafb(block_z, block_y, block_x, 54, 1735, 1735)
            (start_x, start_y, start_z) = (start_x / 4 - 156, start_y / 4 - 156, start_z - 29)
            (end_x, end_y, end_z) = (end_x / 4 + 156 + 1, end_y / 4 + 156 + 1, end_z + 29 + 1)
            print('begin dowload: ', file_name)
            try:
                volume = fafb_v14[start_x:end_x, start_y:end_y, start_z:end_z]
            except:
                print('fail to download', file_name)
            os.makedirs(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14', folder_name))
            for i in range(84):
                im = Image.fromarray(volume[:, :, i, 0])
                im.save(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14',
                                     os.path.join(folder_name, str(i).zfill(4) + '.png')))
    else:
        df = pd.read_csv('/braindat/lab/liusl/flywire/block_data/30_percent.csv')
        for i in df.index:
            if i>4000: break
            file_name = df['block'][i]
            folder_name = file_name.split('.')[0]
            if not os.path.exists(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14', folder_name)):
                [block_x, block_y, block_z] = file_name.split('_')[1:]
                block_x = int(block_x)
                block_y = int(block_y)
                block_z = int(block_z.split('.')[0])
                (start_x, start_y, start_z) = xpblock_to_fafb(block_z, block_y, block_x, 29, 0, 0)
                (end_x, end_y, end_z) = xpblock_to_fafb(block_z, block_y, block_x, 54, 1735, 1735)
                (start_x, start_y, start_z) = (start_x / 4 - 156, start_y / 4 - 156, start_z - 29)
                (end_x, end_y, end_z) = (end_x/4+156+1, end_y/4+156+1, end_z+29+1)
                print('begin dowload: ', file_name)
                try:
                    volume = fafb_v14[start_x:end_x, start_y:end_y, start_z:end_z]
                except:
                    print('fail to download', file_name)
                    continue
                os.makedirs(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14', folder_name))
                for i in range(84):
                    im = Image.fromarray(volume[:, :, i, 0])
                    im.save(os.path.join('/braindat/lab/liusl/flywire/block_data/fafbv14', os.path.join(folder_name, str(i).zfill(4)+'.png')))


def zfill_folders(path):
    paths = os.listdir(path)
    for p in paths:
        zfill_all(os.path.join(path, p))


if __name__ == '__main__':
    # zfill_folders(r'H:\fafbv14')
    download()