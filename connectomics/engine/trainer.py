from __future__ import print_function, division

import random
from typing import Optional
import warnings

import os
import time
import math
import GPUtil
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode
import einops
import dill
import pandas as pd
import tifffile as tf

import torch
import torchvision.transforms as transforms
from cloudvolume import CloudVolume
# from torch.cuda.amp import autocast, GradScaler

from .solver import *
from ..model import *
from ..utils.monitor import build_monitor
from plyfile import PlyData
from ..utils.evaluate import visualize
from torch.cuda.amp import autocast, GradScaler
from ..data.augmentation import build_train_augmentor, TestAugmentor
from ..data.dataset import build_dataloader, get_dataset, ConnectorDataset
from ..data.dataset.build import _get_file_list, _make_path_list
from ..data.utils import build_blending_matrix, writeh5, get_connection_distance, get_connection_ranking, pca_emb, \
    readh5
from ..data.utils import get_padsize, array_unpad, readvol, patch_rand_drop, stat_biological_recall, select_points, \
    get_crop_index

DEBUG = 0


class Trainer(object):
    r"""Trainer class for supervised learning.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``). Default: ``'train'``
        rank (int, optional): node rank for distributed training. Default: `None`
        checkpoint (str, optional): the checkpoint file to be loaded. Default: `None`
    """

    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None,
                 checkpoint: Optional[str] = None,
                 cfg_image_model: CfgNode = None,
                 checkpoint_image_model: Optional[str] = None, ):

        assert mode in ['train', 'test']
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        self.mode = mode
        self.rank = rank
        self.is_main_process = rank is None or rank == 0
        self.inference_singly = (mode == 'test') and cfg.INFERENCE.DO_SINGLY

        self.model = build_model(self.cfg, self.device, rank)
        if self.mode == 'train':
            self.optimizer = build_optimizer(self.cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
            self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
            self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
            self.update_checkpoint(checkpoint)

            # stochastic weight averaging
            if self.cfg.SOLVER.SWA.ENABLED:
                self.swa_model, self.swa_scheduler = build_swa_model(
                    self.cfg, self.model, self.optimizer)

            self.augmentor = build_train_augmentor(self.cfg)
            self.criterion = Criterion.build_from_cfg(self.cfg, self.device)
            if self.is_main_process:
                self.monitor = build_monitor(self.cfg)
                self.monitor.load_info(self.cfg, self.model)

            self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
            self.total_time = 0
            self.chunk_time = 0
            self.chunk_total_time = 0
        else:
            self.update_checkpoint(checkpoint)
            # build test-time augmentor and update output filename
            self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
            if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME
                self.test_filename = self.augmentor.update_name(
                    self.test_filename)

        self.dataset, self.dataloader, self.dataset_val = None, None, None
        if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly and not self.cfg.DATASET.DO_MULTI_VOLUME:
            self.dataloader = build_dataloader(
                self.cfg, self.augmentor, self.mode, rank=rank)
            self.dataloader = iter(self.dataloader)
            if self.mode == 'train' and cfg.DATASET.VAL_IMAGE_NAME is not None:
                self.val_loader = build_dataloader(
                    self.cfg, None, mode='val', rank=rank)
        if self.cfg.DATASET.MORPHOLOGY_DATSET and self.cfg.MODEL.IN_PLANES > 4:
            assert cfg_image_model is not None
            self.cfg_image_model = cfg_image_model
            self.image_model = build_model(cfg_image_model, self.device, rank)
            self.update_checkpoint(checkpoint_image_model, is_image_model=True)
            self.image_model.eval()

    def train(self):
        r"""Training function of the trainer class.
        """
        self.model.train()
        self.total_time = 0
        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            # load data
            sample = next(self.dataloader)
            volume = sample.out_input
            if self.cfg.DATASET.MORPHOLOGY_DATSET:
                volume, embedding = self.get_morph_input(volume)
            target, weight = sample.out_target_l, sample.out_weight_l

            self.data_time = time.perf_counter() - self.start_time

            # prediction
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                if self.cfg.MODEL.EMBED_REDUCTION is None:
                    volume = volume.contiguous().to(self.device, non_blocking=True)
                    pred = self.model(volume)
                else:
                    pred = self.model(volume, embedding)
                loss, losses_vis = self.criterion(pred, target, weight)

            self._train_misc(loss, pred, volume, target, weight, iter_total, losses_vis)

            # from sklearn.decomposition import PCA
            # volume_save = np.asarray(volume.clone().detach().cpu())
            # x = (volume_save[0, 0, :] - volume_save[0, 0, :].min()) * 240
            # tf.imsave('volume.tif', x.astype(np.uint8))
            # x = np.asarray(emb2rgb(pred))
            # x = (x[0, :] - x[0, :].min()) / (x[0, :].max() - x[0, :].min()) * 240
            # tf.imsave('volume.tif', x.transpose(1, 0, 2, 3).astype(np.uint8))
            # volume_save = np.asarray(target[0].clone().detach().cpu())
            # labels = np.unique(volume_save[0, 1, 0, :])
            # mask = volume_save[0, 1, 0, :] == labels[1]
            # tf.imsave('target1.tif', (mask*240).astype(np.uint8))

        self.maybe_save_swa_model()

    def _train_misc(self, loss, pred, volume, target, weight,
                    iter_total, losses_vis):
        self.backward_pass(loss)  # backward pass

        # logging and update record
        if hasattr(self, 'monitor'):
            do_vis = self.monitor.update(iter_total, loss, losses_vis,
                                         self.optimizer.param_groups[0]['lr'])
            if do_vis:
                if self.cfg.DATASET.MORPHOLOGY_DATSET:
                    if torch.cuda.is_available():
                        GPUtil.showUtilization(all=True)
                    # self.monitor.plot_3d(pred, target, volume, iter_total, name='train')
                else:
                    self.monitor.visualize(
                        volume, target, pred, weight, iter_total)
                    if torch.cuda.is_available():
                        GPUtil.showUtilization(all=True)

        # Save model
        if (iter_total + 1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
            self.save_checkpoint(iter_total)

        if (iter_total + 1) % self.cfg.SOLVER.ITERATION_VAL == 0:
            self.validate(iter_total)

        # update learning rate
        self.maybe_update_swa_model(iter_total)
        self.scheduler_step(iter_total, loss)
        if self.cfg.SOLVER.DECAY_LOSS_WEIGHT:
            self.criterion.update_weight(iter_total)

        if self.is_main_process:
            self.iter_time = time.perf_counter() - self.start_time
            self.total_time += self.iter_time
            avg_iter_time = self.total_time / (iter_total + 1 - self.start_iter)
            est_time_left = avg_iter_time * \
                            (self.total_iter_nums + self.start_iter - iter_total - 1) / 3600.0
            info = [
                '[Iteration %05d]' % iter_total, 'Data time: %.4fs,' % self.data_time,
                'Iter time: %.4fs,' % self.iter_time, 'Avg iter time: %.4fs,' % avg_iter_time,
                'Time Left %.2fh.' % est_time_left]
            print(' '.join(info))

        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del volume, target, pred, weight, loss, losses_vis

    def validate(self, iter_total):
        r"""Validation function of the trainer class.
        """
        if not hasattr(self, 'val_loader'):
            return

        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            TP_total = 0
            FP_total = 0
            TN_total = 0
            FN_total = 0
            distance_pos_list = []
            distance_neg_list = []
            classification_list = []
            for i, sample in enumerate(self.val_loader):
                volume = sample.out_input
                target, weight = sample.out_target_l, sample.out_weight_l
                if self.cfg.DATASET.MORPHOLOGY_DATSET:
                    volume, embedding = self.get_morph_input(volume)
                # prediction
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    if self.cfg.MODEL.EMBED_REDUCTION is None:
                        volume = volume.contiguous().to(self.device, non_blocking=True)
                        pred = self.model(volume)
                    else:
                        pred = self.model(volume, embedding)
                    loss, _ = self.criterion(pred, target, weight)
                    val_loss += loss.data
                if self.cfg.DATASET.CONNECTOR_DATSET and not self.cfg.DATASET.MORPHOLOGY_DATSET:
                    distance_pos, distance_neg, classification = get_connection_distance(pred, target)
                    distance_neg_list.append(distance_neg)
                    distance_pos_list.append(distance_pos)
                    classification_list = classification_list + classification
                if self.cfg.DATASET.MORPHOLOGY_DATSET:
                    TP = sum(torch.logical_and(pred.detach().cpu() > 0.5, target[0] == 1))
                    TP_total = TP_total + TP
                    FP = sum(torch.logical_and(pred.detach().cpu() > 0.5, target[0] == 0))
                    FP_total = FP_total + FP
                    FN = sum(torch.logical_and(pred.detach().cpu() < 0.5, target[0] == 1))
                    FN_total = FN_total + FN
        name = self.dataset_val.volume_sample.split('/')[-1]
        if hasattr(self, 'monitor'):
            self.monitor.logger.log_tb.add_scalar(
                '%s_Validation_Loss' % name, val_loss, iter_total)
            if not self.cfg.DATASET.MORPHOLOGY_DATSET:
                self.monitor.visualize(volume, target, pred,
                                       weight, iter_total, suffix='Val')
            else:
                recall = TP_total / (TP_total + FN_total)
                accuracy = TP_total / (TP_total + FP_total)
                if DEBUG:
                    row = pd.DataFrame([{'block': self.dataset_val.dataset.connector_path.split('/')[-1],
                                         'connector_num': len(self.dataset_val.dataset), 'recall': recall.item(),
                                         'acc': accuracy.item()}])
                    row.to_csv('/braindat/lab/liusl/flywire/block_data/v2/morph_performence_2.csv', mode='a',
                               header=False, index=False)
                print(recall)
                print(accuracy)
                self.monitor.logger.log_tb.add_scalar(
                    '%s_Validation_classifaction_recall' % name, recall, iter_total)
                self.monitor.logger.log_tb.add_scalar(
                    '%s_Validation_classifaction_accuracy' % name, accuracy, iter_total)
                # self.monitor.plot_3d(pred, target, volume, iter_total, name='val', pos_data=sample.pos)
            if self.cfg.DATASET.CONNECTOR_DATSET and not self.cfg.DATASET.MORPHOLOGY_DATSET:
                self.monitor.plot_distance(distance_pos_list, distance_neg_list, iter_total, name=name)
                accuracy = sum(classification_list) / len(classification_list)
                self.monitor.logger.log_tb.add_scalar(
                    '%s_Validation_classifaction' % name, accuracy, iter_total)

        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = val_loss

        if val_loss < self.best_val_loss and not hasattr(self, 'val_loss_total'):
            self.best_val_loss = val_loss
            self.save_checkpoint(iter_total, is_best=True)
        if hasattr(self, 'val_loss_total'):
            self.val_loss_total = self.val_loss_total + val_loss
        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del pred, loss, val_loss

        # model.train() only called at the beginning of Trainer.train().
        self.model.train()

    def test(self):
        r"""Inference function of the trainer class.
        """
        self.model.eval() if self.cfg.INFERENCE.DO_EVAL else self.model.train()
        print("Total number of batches: ", len(self.dataloader))

        start = time.perf_counter()
        with torch.no_grad():
            for i, sample in enumerate(self.dataloader):
                print('progress: %d/%d batches, total time %.2fs' %
                      (i + 1, len(self.dataloader), time.perf_counter() - start))

                pos, volume, ffn_label, seg_start, candidates = sample.pos, sample.out_input, sample.ffn_label, sample.seg_start, sample.candidates
                volume = volume.to(self.device, non_blocking=True)
                output = self.augmentor(self.model, volume)
                ranking = get_connection_ranking(output, ffn_label, seg_start, candidates)

                if torch.cuda.is_available() and i % 50 == 0:
                    GPUtil.showUtilization(all=True)

        end = time.perf_counter()
        print("Prediction time: %.2fs" % (end - start))

    def get_test_dataset(self):
        r"""Inference function of the trainer class.
        """
        self.model.eval() if self.cfg.INFERENCE.DO_EVAL else self.model.train()
        print("Total number of batches: ", len(self.dataloader))

        start = time.perf_counter()
        with torch.no_grad():
            for i, sample in enumerate(self.dataloader):
                print('progress: %d/%d batches, total time %.2fs' %
                      (i + 1, len(self.dataloader), time.perf_counter() - start))

                if torch.cuda.is_available() and i % 50 == 0:
                    GPUtil.showUtilization(all=True)

        end = time.perf_counter()
        print("Prediction time: %.2fs" % (end - start))

    def test_embededge(self, visualize_csv_path=None):
        r"""Inference function of the trainer class.
        """
        self.model.eval() if self.cfg.INFERENCE.DO_EVAL else self.model.train()
        print("Total number of batches: ", len(self.dataloader))
        TP_total = 0
        FP_total = 0
        TN_total = 0
        FN_total = 0
        start = time.perf_counter()
        with torch.no_grad():
            for i, sample in enumerate(self.dataloader):
                print('progress: %d/%d batches, total time %.2fs' %
                      (i + 1, len(self.dataloader), time.perf_counter() - start))

                volume = sample.out_input
                target = torch.tensor(np.expand_dims(np.asarray(sample.seg_start), axis=1))
                ids = sample.candidates
                volume, embedding = self.get_morph_input(volume)
                # visualize(volume, sample.out_input, index=22, mask=True)
                # prediction
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    if self.cfg.MODEL.EMBED_REDUCTION is None:
                        volume = volume.contiguous().to(self.device, non_blocking=True)
                        pred = self.model(volume)
                    else:
                        pred = self.model(volume, embedding)
                pred_record = pred.detach().cpu()
                TP = sum(torch.logical_and(pred_record > 0.5, target == 1))
                TP_total = TP_total + TP
                FP = sum(torch.logical_and(pred_record > 0.5, target == 0))
                FP_total = FP_total + FP
                FN = sum(torch.logical_and(pred_record < 0.5, target == 1))
                FN_total = FN_total + FN
                if visualize_csv_path is not None:
                    for idx in range(len(target)):
                        row = pd.DataFrame(
                            [{'node0_segid': int(ids[idx][0]), 'node1_segid': int(ids[idx][1]),
                              'target': int(target[idx] == 1),
                              'prediction': int((pred_record > 0.5)[idx]), 'value': pred_record[idx].item()}])
                        row.to_csv(visualize_csv_path, mode='a', header=False, index=False)
                if torch.cuda.is_available() and i % 50 == 0:
                    GPUtil.showUtilization(all=True)

        recall = TP_total / (TP_total + FN_total)
        accuracy = TP_total / (TP_total + FP_total)
        end = time.perf_counter()
        print("Prediction time: %.2fs" % (end - start))
        return recall, accuracy

    def test_one_neuron(self):
        dir_name = _get_file_list(self.cfg.DATASET.INPUT_PATH)
        block_name = _make_path_list(self.cfg, dir_name, None, self.rank)
        block_image_path = '/braindat/lab/liusl/flywire/block_data/fafbv14'
        vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0')
        num_file = len(block_name)
        start_idx = self.cfg.INFERENCE.DO_SINGLY_START_INDEX
        for i in range(start_idx, num_file):
            print('Processing block: ', block_name[i])
            block_index = block_name[i].split('/')[-1].split('.')[0]
            block_xyz = block_index.split('_')[1:]
            img_volume_path = os.path.join(block_image_path, block_index, '*.png')
            volume = readvol(img_volume_path)
            volume = [volume[12:70, :, :]]
            # volume = volume[29:55, 156:-156, 156:-156]
            cord_start = self.xpblock_to_fafb(int(block_xyz[2]), int(block_xyz[1]), int(block_xyz[0]), 28, 0, 0)
            cord_end = self.xpblock_to_fafb(int(block_xyz[2]), int(block_xyz[1]), int(block_xyz[0]), 53, 1735, 1735)
            volume_ffn1 = vol_ffn1[cord_start[0] / 4 - 156:cord_end[0] / 4 + 156 + 1,
                          cord_start[1] / 4 - 156:cord_end[1] / 4 + 156 + 1, cord_start[2] - 16:cord_end[2] + 16 + 1]
            # volume_ffn1 = self.vol_ffn1[cord_start[0] / 4:cord_end[0] / 4 + 1,
            #               cord_start[1] / 4:cord_end[1] / 4 + 1, cord_start[2]:cord_end[2] + 1]
            volume_ffn1 = np.transpose(volume_ffn1.squeeze(), (2, 0, 1))
            dataset = ConnectorDataset(connector_path=block_name[i], volume=volume, vol_ffn1=volume_ffn1,
                                       mode=self.mode,
                                       iter_num=-1, model_input_size=self.cfg.MODEL.INPUT_SIZE,
                                       sample_volume_size=self.cfg.MODEL.INPUT_SIZE)
            self.dataloader = build_dataloader(self.cfg, self.augmentor, self.mode, dataset, self.rank)
            self.dataloader = iter(self.dataloader)

            digits = int(math.log10(num_file)) + 1
            self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME + \
                                 '_' + str(i).zfill(digits) + '.h5'
            self.test_filename = self.augmentor.update_name(
                self.test_filename)

            self.test()

    def test_biological(self):
        volume = readh5(self.cfg.DATASET.IMAGE_NAME)
        volume_ffn1 = readh5((self.cfg.DATASET.IMAGE_NAME).replace('connector', 'ffn'))
        mapping = readh5(self.cfg.DATASET.LABEL_NAME, 'original')
        edges = readh5((self.cfg.DATASET.LABEL_NAME).replace('mapping.h5', '.h5').replace('segmentationsfafb', 'fafb'))
        csv_path = (self.cfg.DATASET.IMAGE_NAME).replace('.h5', '.csv')
        result_csv_path = csv_path.replace('.csv', '_result.csv')

        # get csv file
        # for edge in tqdm(edges):
        #     seg_start = mapping[edge[3] - 1]
        #     seg_candidate = mapping[edge[4] - 1]
        #     cord = np.asarray([edge[0], edge[1], edge[2]]) - np.asarray([8, 64, 64])
        #     upper_bound = np.asarray([volume.shape[2], volume.shape[1], volume.shape[0]]) - np.asarray([16, 128, 128])
        #     cord = [np.clip(cord[0], 0, upper_bound[0]), np.clip(cord[1], 0, upper_bound[1]), np.clip(cord[2], 0, upper_bound[2])]
        #     row = pd.DataFrame(
        #         [{'node0_segid': int(seg_start), 'node1_segid': int(seg_candidate), 'cord': cord, 'target': -1,
        #           'prediction': -1}])
        #     row.to_csv(csv_path, mode='a', header=False, index=False)
        # stat_biological_recall(csv_path)

        dataset = ConnectorDataset(connector_path=csv_path, volume=volume, vol_ffn1=volume_ffn1,
                                   mode=self.mode,
                                   iter_num=-1, model_input_size=self.cfg.MODEL.INPUT_SIZE,
                                   sample_volume_size=self.cfg.MODEL.INPUT_SIZE)
        self.dataloader = build_dataloader(self.cfg, self.augmentor, self.mode, dataset, self.rank)
        self.dataloader = iter(self.dataloader)
        start = time.perf_counter()
        with torch.no_grad():
            for i, sample in enumerate(self.dataloader):
                print('progress: %d/%d batches, total time %.2fs' %
                      (i + 1, len(self.dataloader), time.perf_counter() - start))

                volume = sample.out_input
                target = torch.tensor(np.expand_dims(np.asarray(sample.seg_start), axis=1))
                ids = sample.candidates
                volume, embedding = self.get_morph_input(volume)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    volume = volume.contiguous().to(self.device, non_blocking=True)
                    pred = self.model(volume)
                for idx in range(len(target)):
                    row = pd.DataFrame(
                        [{'node0_segid': int(ids[idx][0]), 'node1_segid': int(ids[idx][1]),
                          'target': int(target[idx] == 1), 'value': pred.detach().cpu()[idx].item()}])
                    row.to_csv(result_csv_path, mode='a', header=False, index=False)

    # -----------------------------------------------------------------------------
    # Misc functions
    # -----------------------------------------------------------------------------

    def backward_pass(self, loss):
        if self.cfg.MODEL.MIXED_PRECESION:
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            self.scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

        else:  # standard backward pass
            loss.backward(torch.ones_like(loss))
            self.optimizer.step()

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        r"""Save the model checkpoint.
        """
        if self.is_main_process:
            print("Save model checkpoint at iteration ", iteration)
            state = {'iteration': iteration + 1,
                     # Saving DataParallel or DistributedDataParallel models
                     'state_dict': self.model.module.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict()}

            # Saves checkpoint to experiment directory
            filename = 'checkpoint_%05d.pth' % (iteration + 1)
            if is_best:
                filename = 'checkpoint_best.pth'
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def update_checkpoint(self, checkpoint: Optional[str] = None, is_image_model=False):
        r"""Update the model with the specified checkpoint file path.
        """
        if checkpoint is None:
            return
        if is_image_model:
            model = self.image_model
        else:
            model = self.model
        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint, map_location='cpu')
        print('checkpoints: ', checkpoint.keys())

        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = update_state_dict(
                self.cfg, pretrained_dict, mode=self.mode)
            model_dict = model.module.state_dict()  # nn.DataParallel

            # show model keys that do not match pretrained_dict
            if not model_dict.keys() == pretrained_dict.keys():
                warnings.warn("Module keys in model.state_dict() do not exactly "
                              "match the keys in pretrained_dict!")
                for key in model_dict.keys():
                    if not key in pretrained_dict:
                        print(key)

            # 1. filter out unnecessary keys by name
            pretrained_dict = {k: v for k,
                                        v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict (if size match)
            for param_tensor in pretrained_dict:
                if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                    model_dict[param_tensor] = pretrained_dict[param_tensor]
                else:
                    warnings.warn("Parameter size in model.state_dict() do not exactly "
                                  "match the keys in pretrained_dict!")
            # 3. load the new state dict
            model.module.load_state_dict(model_dict)  # nn.DataParallel
            # self.model.module.load_from(checkpoint)

        if self.mode == 'train' and not self.cfg.SOLVER.ITERATION_RESTART and not is_image_model:
            if hasattr(self, 'optimizer') and 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if hasattr(self, 'lr_scheduler') and 'lr_scheduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            if hasattr(self, 'start_iter') and 'iteration' in checkpoint.keys():
                self.start_iter = checkpoint['iteration']

    def maybe_save_swa_model(self):
        if not hasattr(self, 'swa_model'):
            return

        if self.cfg.MODEL.NORM_MODE in ['bn', 'sync_bn']:  # update bn statistics
            for _ in range(self.cfg.SOLVER.SWA.BN_UPDATE_ITER):
                sample = next(self.dataloader)
                volume = sample.out_input
                volume = volume.to(self.device, non_blocking=True)
                pred = self.swa_model(volume)

        # save swa model
        if self.is_main_process:
            print("Save SWA model checkpoint.")
            state = {'state_dict': self.swa_model.module.state_dict()}
            filename = 'checkpoint_swa.pth.tar'
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def maybe_update_swa_model(self, iter_total):
        if not hasattr(self, 'swa_model'):
            return

        swa_start = self.cfg.SOLVER.SWA.START_ITER
        swa_merge = self.cfg.SOLVER.SWA.MERGE_ITER
        if iter_total >= swa_start and iter_total % swa_merge == 0:
            self.swa_model.update_parameters(self.model)

    def scheduler_step(self, iter_total, loss):
        if hasattr(self, 'swa_scheduler') and iter_total >= self.cfg.SOLVER.SWA.START_ITER:
            self.swa_scheduler.step()
            return

        if self.cfg.SOLVER.LR_SCHEDULER_NAME == 'ReduceLROnPlateau':
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

    # -----------------------------------------------------------------------------
    # Chunk processing for TileDataset
    # -----------------------------------------------------------------------------
    def run_chunk(self, mode: str):
        r"""Run chunk-based training and inference for large-scale datasets.
        """
        self.dataset = get_dataset(self.cfg, self.augmentor, mode)
        if mode == 'train':
            num_chunk = self.total_iter_nums // self.cfg.DATASET.DATA_CHUNK_ITER
            self.total_iter_nums = self.cfg.DATASET.DATA_CHUNK_ITER
            for chunk in range(num_chunk):
                self.dataset.updatechunk()
                self.dataloader = build_dataloader(self.cfg, self.augmentor, mode,
                                                   dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)
                print('start train for chunk %d' % chunk)
                self.train()
                print('finished train for chunk %d' % chunk)
                self.start_iter += self.cfg.DATASET.DATA_CHUNK_ITER
                del self.dataloader
            return

        # inference mode
        num_chunk = len(self.dataset.chunk_ind)
        print("Total number of chunks: ", num_chunk)
        for chunk in range(num_chunk):
            self.dataset.updatechunk(do_load=False)
            self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME + \
                                 '_' + self.dataset.get_coord_name() + '.h5'
            self.test_filename = self.augmentor.update_name(
                self.test_filename)
            if not os.path.exists(os.path.join(self.output_dir, self.test_filename)):
                self.dataset.loadchunk()
                self.dataloader = build_dataloader(self.cfg, self.augmentor, mode,
                                                   dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)
                self.test()

    def run_multivolume(self, mode: str, rank=None):
        r"""Run multi volume training for mixed datasets.
        """
        self.dataset = get_dataset(self.cfg, self.augmentor, mode, rank=rank)
        if mode == 'train':
            if self.cfg.DATASET.VAL_PATH is not None:
                self.dataset_val = get_dataset(self.cfg, self.augmentor, mode='val', rank=rank)
            num_chunk = self.total_iter_nums // self.cfg.DATASET.DATA_CHUNK_ITER
            self.total_iter_nums = self.cfg.DATASET.DATA_CHUNK_ITER
            for chunk in range(num_chunk):
                # if self.start_iter % self.cfg.SOLVER.ITERATION_VAL == 0:  # ITERATION_VAL should be k*DATA_CHUNK_ITER
                #     self.val_loader = build_dataloader(self.cfg, None, mode='val', dataset=self.dataset.dataset)
                self.chunk_time = time.time()
                self.dataset.updatechunk()
                self.dataloader = build_dataloader(self.cfg, self.augmentor, mode,
                                                   dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)

                print('chunk loading time:', time.time() - self.chunk_time)
                print('rank:', rank)
                print('start train for chunk %d' % chunk)
                self.train()
                print('finished train for chunk %d' % chunk)
                self.start_iter += self.cfg.DATASET.DATA_CHUNK_ITER
                del self.dataloader
                print('chunk time:', time.time() - self.chunk_time)
                # ITERATION_VAL should be k*DATA_CHUNK_ITER, and in main process
                if self.is_main_process:
                    if self.dataset_val is not None and self.start_iter % self.cfg.SOLVER.ITERATION_VAL == 0:
                        self.dataset_val.volume_done = []
                        self.val_loss_total = 0

                        while len(self.dataset_val.volume_done) < len(self.dataset_val.volume_path):
                            self.dataset_val.updatechunk()
                            self.val_loader = build_dataloader(self.cfg, None, mode='val',
                                                               dataset=self.dataset_val.dataset)
                            self.val_loader = iter(self.val_loader)
                            self.validate(self.start_iter)
                            del self.val_loader
                        if not hasattr(self, 'best_val_loss_total'):
                            self.best_val_loss_total = self.val_loss_total
                        self.monitor.logger.log_tb.add_scalar(
                            'Total_Validation_Loss', self.val_loss_total, self.start_iter)
                        print('Total_Validation_Loss', self.val_loss_total)
                        if self.val_loss_total < self.best_val_loss_total:
                            self.best_val_loss_total = self.val_loss_total
                            self.save_checkpoint(chunk, is_best=True)
        else:
            # inference for EmbedEdgeNetwork
            num_chunk = len(self.dataset.volume_path)
            rec_list = []
            acc_list = []
            print("Total number of chunks: ", num_chunk)
            result_path = os.path.join(self.cfg.DATASET.OUTPUT_PATH, 'block_result.csv')
            if not os.path.exists(os.path.join(self.cfg.DATASET.OUTPUT_PATH, 'predictions')):
                os.makedirs(os.path.join(self.cfg.DATASET.OUTPUT_PATH, 'predictions'))
            block_path = self.dataset.volume_path.copy()
            random.seed(time.time())
            random.shuffle(block_path)
            for chunk in range(num_chunk):
                csv_path = block_path[chunk]
                block_name = csv_path.split('/')[-1]
                visualize_csv_path = os.path.join(self.cfg.DATASET.OUTPUT_PATH, 'predictions', block_name)
                if os.path.exists(visualize_csv_path):
                    print('block %s is already tested.' % block_name)
                    continue
                self.dataset.updatechunk_given_path(csv_path)
                # self.dataset.updatechunk()
                csv_path = self.dataset.dataset.connector_path.replace('30_percent_test_3000',
                                                                       '30_percent_test_3000_reformat')
                csv_path = csv_path.replace('30_percent_train_1000', '30_percent_train_1000_reformat')
                if not os.path.exists(csv_path):
                    print('#W Test data do not exist, extracting.')
                    self.dataset.dataset.make_test_data = True
                    self.dataloader = build_dataloader(self.cfg, None, mode, dataset=self.dataset.dataset)
                    self.dataloader = iter(self.dataloader)
                    self.get_test_dataset()
                else:
                    self.dataloader = build_dataloader(self.cfg, None, mode, dataset=self.dataset.dataset)
                    self.dataloader = iter(self.dataloader)
                    rec, acc = self.test_embededge(visualize_csv_path)
                    rec_list.append(rec.item())
                    acc_list.append(acc.item())
                    row = pd.DataFrame(
                        [{'block_name': block_name, 'recall': rec.item(), 'accuracy': acc.item()}])
                    row.to_csv(result_path, mode='a', header=False, index=False)
            row = pd.DataFrame(
                [{'block_name': 'average', 'recall': np.mean(np.array(rec_list)),
                  'accuracy': np.mean(np.asarray(acc_list))}])
            row.to_csv(result_path, mode='a', header=False, index=False)
            return

    def get_pc_feature(self, mode: str, rank=None):
        r"""Run multi volume training for mixed datasets.
        """
        self.dataset = get_dataset(self.cfg, self.augmentor, mode, rank=rank)
        num_chunk = len(self.dataset.volume_path)
        print("Total number of chunks: ", num_chunk)
        half_patch_size = np.asarray([3, 3, 1])
        # pc_path_pos = '/braindat/lab/daiyi.zhu/flywire/block_data/v2/point_cloud/train_fps/pos'
        # pc_path_neg = '/braindat/lab/daiyi.zhu/flywire/block_data/v2/point_cloud/train_fps/neg'
        # pc_result_root_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/train_feature/'
        pc_path_pos = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps/pos'
        pc_path_neg = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_fps/neg'
        pc_result_root_path = '/braindat/lab/liusl/flywire/block_data/v2/point_cloud/test_feature/'
        out_put_dir = self.cfg.DATASET.OUTPUT_PATH.split('/')[-1]
        pc_result_root_path = os.path.join(pc_result_root_path, out_put_dir)
        patch_size = np.asarray(
            [self.cfg.MODEL.INPUT_SIZE[1], self.cfg.MODEL.INPUT_SIZE[2], self.cfg.MODEL.INPUT_SIZE[0]])
        block_path = self.dataset.volume_path.copy()
        random.seed(time.time())
        random.shuffle(block_path)
        for chunk in range(num_chunk):
            csv_path = block_path[chunk]
            block_name = csv_path.split('/')[-1].split('.')[0]
            # block_name = 'connector_18_10_121'
            pc_ply_path_pos = os.path.join(pc_path_pos, block_name)
            pc_ply_path_neg = os.path.join(pc_path_neg, block_name)
            pc_result_path = os.path.join(pc_result_root_path, block_name)
            if os.path.exists(pc_ply_path_neg) and os.path.exists(pc_ply_path_pos) and not os.path.exists(pc_result_path):
                os.makedirs(pc_result_path)
                csv_list = pd.read_csv(csv_path, header=None)
                self.dataset.updatechunk_given_path(csv_path)
                self.dataloader = build_dataloader(self.cfg, None, mode, dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)
                block_start_cord = self.dataset.start_cord
                start = time.perf_counter()
                with torch.no_grad():
                    for i, sample in enumerate(self.dataloader):
                        print('progress: %d/%d batches, total time %.2fs' %
                              (i + 1, len(self.dataloader), time.perf_counter() - start))
                        volume = sample.out_input
                        seg_ids = sample.candidates
                        volume, _ = self.get_morph_input(volume, sample.pos)
                        label = sample.seg_start
                        for idx in range(len(seg_ids)):
                            name = str(seg_ids[idx][0]) + '_' + str(seg_ids[idx][1]) + '.ply'
                            if label[idx] == 0:
                                pc_ply_path = pc_ply_path_neg
                            else:
                                pc_ply_path = pc_ply_path_pos
                            if os.path.exists(os.path.join(pc_ply_path, name)):
                                pc = PlyData.read(os.path.join(pc_ply_path, name))
                                patch_cord = np.asarray(block_start_cord) + np.asarray(
                                    [sample.pos[idx][2] * 4, sample.pos[idx][3] * 4, sample.pos[idx][1]])
                                bbox = [list(patch_cord), list(patch_cord + patch_size * [4, 4, 1])]
                                x = pc.elements[0].data['x']
                                y = pc.elements[0].data['y']
                                z = pc.elements[0].data['z']
                                ids = pc.elements[0].data['id']
                                cords = np.transpose(np.asarray([x, y, z]) / np.expand_dims(np.asarray([4, 4, 40]), axis=1),
                                                     [1, 0])
                                embeddings = np.zeros([len(x), 16])
                                in_box_indexes = np.where(select_points(bbox, cords))[0]
                                for in_box_index in in_box_indexes:
                                    cord_in_patch = np.round((cords[in_box_index] - patch_cord) / [4, 4, 1]).astype(
                                        np.int32)
                                    [x_bound, y_bound, z_bound] = get_crop_index(cord_in_patch, half_patch_size, patch_size)
                                    local_embed = volume[idx, 3:, z_bound[0]:z_bound[1], x_bound[0]:x_bound[1],
                                                  y_bound[0]:y_bound[1]]
                                    local_mask = volume[idx, ids[in_box_index], z_bound[0]:z_bound[1],
                                                 x_bound[0]:x_bound[1], y_bound[0]:y_bound[1]] == 1
                                    if not local_mask.any():
                                        continue
                                    mean_embed = torch.mean(local_embed[:, local_mask], dim=1)
                                    embeddings[in_box_index, :] = mean_embed.detach().cpu().numpy()
                                writeh5(os.path.join(pc_result_path, name.replace('.ply', '.h5')), embeddings)

        return

    def xpblock_to_fafb(self, z_block, y_block, x_block, z_coo=0, y_coo=0, x_coo=0):
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

    def get_morph_input(self, volume, pos=None):
        volume_image = volume[:, 0, :, :, :]
        volume_morph = volume[:, 1:, :, :, :]
        if pos is not None:
            pos_record = pos[0]
        if self.cfg.MODEL.IN_PLANES > 4:
            if self.cfg.MODEL.MASK_EMBED:
                morph_dim = 2
            else:
                morph_dim = 3
            embed_dim = self.cfg.MODEL.IN_PLANES - morph_dim
            with torch.no_grad():
                # image_batch_size = self.cfg_image_model.SOLVER.SAMPLES_PER_BATCH  # must be 1
                image_batch_size = 1
                num_batch = int(np.ceil(volume_image.shape[0] / image_batch_size))
                volume_embedding_list = []

                for i in range(num_batch):
                    if pos is not None:
                        if np.all(pos[i] == pos_record) and i > 0:
                            volume_embedding_list.append(volume_embedding)
                            continue
                        pos_record = pos[i]
                    input_image = volume_image[i * image_batch_size:(i + 1) * image_batch_size].to(self.device,
                                                                                                   non_blocking=True)
                    input_image = input_image.unsqueeze(dim=1)
                    volume_embedding = self.image_model(input_image)

                    if self.cfg.MODEL.DROP_MOD:
                        if random.random() < 0.1:
                            volume_embedding[:] = torch.mean(volume_embedding)
                        volume_morph[i, 0, :, :, :] = patch_rand_drop(volume_morph[i, 0, :, :, :].unsqueeze(dim=0))
                        volume_morph[i, 1, :, :, :] = patch_rand_drop(volume_morph[i, 1, :, :, :].unsqueeze(dim=0))
                        volume_morph[i, 2, :, :, :] = volume_morph[i, 0, :, :, :] + volume_morph[i, 1, :, :, :]
                    volume_embedding_list.append(volume_embedding)

                volume_embedding = torch.cat(volume_embedding_list, dim=0)

                if self.cfg.MODEL.EMBED_REDUCTION is not None:
                    return volume_morph.contiguous().to(self.device, non_blocking=True), volume_embedding
                if embed_dim < volume_embedding.shape[1]:
                    volume_embedding = pca_emb(volume_embedding, dim=volume_embedding.shape[1],
                                               n_components=embed_dim).to(self.device, non_blocking=True)
                if self.cfg.MODEL.MASK_EMBED:
                    mask = einops.repeat(volume_morph[:, -1, :, :, :], 'b d h w -> b k d h w', k=embed_dim).to(
                        self.device, non_blocking=True)
                    volume_embedding = mask * volume_embedding
                    volume_input = torch.cat(
                        (volume_morph[:, 0:2, :, :, :].to(self.device, non_blocking=True), volume_embedding), dim=1)
                else:
                    volume_input = torch.cat((volume_morph.to(self.device, non_blocking=True), volume_embedding), dim=1)
        elif self.cfg.MODEL.IN_PLANES == 4:
            volume_input = volume
        else:
            volume_input = volume_morph
        if self.cfg.MODEL.MORPH_INPUT_SIZE is not None:
            resize = torch.nn.Upsample(size=self.cfg.MODEL.MORPH_INPUT_SIZE)
            volume_input = resize(volume_input)
        return volume_input, 0
