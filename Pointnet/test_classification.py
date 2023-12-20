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
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()



def test(model, loader, summary_writer, criterion, step):
    FP_total = 0
    TP_total = 0
    FN_total = 0
    mean_correct = []
    loss_vis = 0
    recall = 0
    accuracy = 0
    classifier = model.eval()
    with torch.no_grad():
        for j, (points, target) in enumerate(loader):

            points = torch.Tensor(points.data.numpy())
            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            target = target.to(pred.device)
            if criterion:
                loss = criterion(pred, target.float(), _)
                loss_vis = loss_vis + loss.item()

            TP = sum(torch.logical_and((pred.detach().cpu() > 0.5).squeeze(), target.cpu() == 1))
            TP_total = TP_total + TP
            FP = sum(torch.logical_and((pred.detach().cpu() > 0.5).squeeze(), target.cpu() == 0))
            FP_total = FP_total + FP
            FN = sum(torch.logical_and((pred.detach().cpu() < 0.5).squeeze(), target.cpu() == 1))
            FN_total = FN_total + FN

        recall = TP_total / (TP_total + FN_total)
        accuracy = TP_total / (TP_total + FP_total)
        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar('test_loss', loss_vis/j, step=step)
                tf.summary.scalar('test_recall', recall, step=step)
                tf.summary.scalar('test_precision', accuracy, step=step)
    return recall, accuracy, loss_vis


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps'

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)

    classifier = model.get_model(feature_channel=1)
    if not args.use_cpu:
        classifier = classifier.cuda()

    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])

    '''TESTING'''
    block_root = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps/block_names'
    result_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps/result.csv'

    test_names = os.listdir(block_root)
    for test_block in test_names:
        test_dataset = FlyTracingDataLoader(root=data_path, split='test', test_path=os.path.join(block_root, test_block), daiyi_datapath=False)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=8)
        with torch.no_grad():
            start = time.perf_counter()
            recall, acc, test_loss = test(classifier, testDataLoader, summary_writer=None, criterion=None, step=0)
            print('progress: block %s total time %.2fs' %
                  (test_block, time.perf_counter() - start))
            log_string('Test Recall: %f, Precision: %f' % (recall, acc))
            row = pd.DataFrame(
                [{'block_name': test_block, 'recall': recall.item(), 'accuracy': acc.item()}])
            row.to_csv(result_path, mode='a', header=False, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
