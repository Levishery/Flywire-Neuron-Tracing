"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import time

import datetime
import logging
import provider
import importlib
import shutil
import tensorflow as tf
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from data_utils.FlyTracingDataLoader import FlyTracingDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--num_gpus', type=int, default=1, help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--num_category', default=2, type=int, help='number classes')
    parser.add_argument('--learning_rate', default=0.002, type=float, help='learning rate in training')
    parser.add_argument('--step_size', default=20, type=float, help='step size for lr decay')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--restart', action='store_true', default=False, help='restart training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--rotate', action='store_true', default=False, help='random rotate aug')
    parser.add_argument('--image_feature', type=str, default=None, help='image feature dir')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, summary_writer, step, prediction_csv):
    classifier = model.eval()
    with torch.no_grad():
        for j, (points, target, name) in enumerate(loader):

            points = torch.Tensor(points.data.numpy())
            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            for i in range(len(name)):
                seg_id0 = int(name[i].split('.')[0].split('_')[0])
                seg_id1 = int(name[i].split('.')[0].split('_')[1])
                if target[i] != -1:
                    row = pd.DataFrame(
                        [{'node0_segid': seg_id0, 'node1_segid': seg_id1,
                          'target': int(target[i] == 1),
                          'prediction': int((pred > 0.5)[i]), 'value': pred[i].item()}])
                    row.to_csv(prediction_csv, mode='a', header=False, index=False)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/'

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))
    gpu_device_ids = list(range(args.num_gpus))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.image_feature is not None:
        feature_channel = 17
    else:
        feature_channel = 1
    classifier = model.get_model(feature_channel=feature_channel)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    if not args.use_cpu:
        classifier = classifier.to(device)
        criterion = criterion.to(device)
    classifier = torch.nn.DataParallel(classifier, device_ids=gpu_device_ids)

    if not args.restart:
        try:
            checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
        except:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0
    else:
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_score = 0.0
    best_acc = 0.0
    best_recall = 0.0
    log_dir = str(exp_dir) + '/log'

    '''TESTING'''
    prefix = 'test'
    result_root = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/block_result'
    block_root = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/evaluate_fafb/block_txt'

    test_names = os.listdir(block_root)
    for test_block in test_names[1:]:
        prediction_csv = os.path.join(result_root, test_block.split('.')[0]+'.csv')
        if not os.path.exists(prediction_csv):
            test_dataset = FlyTracingDataLoader(root=data_path, split='test', process_data=args.process_data, use_image_feature=args.image_feature, test_path=os.path.join(block_root, test_block), daiyi_datapath=False, return_name=True, block_name=test_block.split('.')[0], npoints=args.num_point)
            testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                         num_workers=8)
            with torch.no_grad():
                start = time.perf_counter()
                test(classifier, testDataLoader, summary_writer=None, step=0, prediction_csv=prediction_csv)
                print('progress: block %s total time %.2fs' %
                      (test_block, time.perf_counter() - start))

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
