"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import tensorflow as tf
import argparse

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
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


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
        with summary_writer.as_default():
            tf.summary.scalar('test_loss', loss_vis/j, step=step)
            tf.summary.scalar('test_recall', recall, step=step)
            tf.summary.scalar('test_precision', accuracy, step=step)
    return recall, accuracy, loss_vis


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
    data_path = '/data/wspolsl/biological_fafb/pc_data'

    train_dataset = FlyTracingDataLoader(root=data_path, split='train', process_data=args.process_data)
    test_dataset = FlyTracingDataLoader(root=data_path, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))
    gpu_device_ids = list(range(args.num_gpus))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = model.get_model(feature_channel=1)
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
    summary_writer = tf.summary.create_file_writer(log_dir)

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        FP_total = 0
        TP_total = 0
        FN_total = 0
        loss_vis = 0

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        # classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                               smoothing=0.9):
            classifier = classifier.train()
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            if args.rotate:
                points[:, :, 0:3] = provider.rotate_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            pred, trans_feat = classifier(points)
            target = target.to(pred.device)
            loss = criterion(pred, target.float(), trans_feat)
            loss_vis = loss_vis + loss.item()
            # pred_choice = (pred.data>0.5).squeeze()

            TP = sum(torch.logical_and((pred.detach().cpu() > 0.5).squeeze(), target.cpu() == 1))
            TP_total = TP_total + TP
            FP = sum(torch.logical_and((pred.detach().cpu() > 0.5).squeeze(), target.cpu() == 0))
            FP_total = FP_total + FP
            FN = sum(torch.logical_and((pred.detach().cpu() < 0.5).squeeze(), target.cpu() == 1))
            FN_total = FN_total + FN

            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            if batch_id % 100 == 0 and batch_id > 10:
                recall = TP_total / (TP_total + FN_total)
                accuracy = TP_total / (TP_total + FP_total)
                log_string('Train Instance Precision: %f， Recall:%f' % (accuracy, recall))
                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', loss_vis/100, step=batch_id+epoch*len(trainDataLoader))
                    tf.summary.scalar('train_recall', recall, step=batch_id+epoch*len(trainDataLoader))
                    tf.summary.scalar('train_precision', accuracy, step=batch_id + epoch * len(trainDataLoader))
                FP_total = 0
                TP_total = 0
                FN_total = 0
                loss_vis = 0
            if batch_id % 400 == 0 and batch_id > 10:
                with torch.no_grad():
                    test_loss = 0
                    acc = 0
                    recall = 0
                    recall, acc, test_loss = test(classifier, testDataLoader, summary_writer, criterion, step=batch_id+epoch*len(trainDataLoader))

                    if (test_loss >= best_score):
                        best_score = test_loss
                        best_acc = acc
                        best_recall = recall
                        best_epoch = epoch + 1
                        logger.info('Save model...')
                        savepath = str(checkpoints_dir) + '/best_model.pth'
                        log_string('Saving at %s' % savepath)
                        state = {
                            'epoch': best_epoch,
                            'Recall': recall,
                            'Precision': acc,
                            'model_state_dict': classifier.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }
                        torch.save(state, savepath)

                    log_string('Test Recall: %f, Precision: %f' % (recall, acc))
                    log_string('Best Recall: %f, Best Precision: %f' % (best_recall, best_acc))

        recall = TP_total / (TP_total + FN_total)
        accuracy = TP_total / (TP_total + FP_total)
        # train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Precision: %f， Recall:%f' % (accuracy, recall))

        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
