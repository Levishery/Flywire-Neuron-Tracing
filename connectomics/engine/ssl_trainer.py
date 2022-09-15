from __future__ import print_function, division
from typing import Optional
import warnings

import os
import time
import GPUtil
from yacs.config import CfgNode
import numpy as np
from numpy.random import randint

import torch
from torch.nn import functional as F
# from torch.cuda.amp import autocast, GradScaler

from.trainer import Trainer


def patch_rand_drop(x,
                    x_rep=None,
                    max_drop=0.15,
                    max_block_sz=0.25,
                    tolr=0.05):
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (np.ceil(tolr * h), np.ceil(tolr * w), np.ceil(tolr * z))  # mask no smaller than tolr
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty((c, rnd_h - rnd_r,
                                           rnd_w - rnd_c,
                                           rnd_z - rnd_s),
                                          dtype=x.dtype).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / \
                              (torch.max(x_uninitialized) - torch.min(x_uninitialized))
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x

def aug_rand(samples):
    # mask as noise & other image
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(x_aug[i],
                                       x_aug[idx_rnd])
    return x_aug


class Contrast(torch.nn.Module):
    def __init__(self, local_rank,
                 batch_size,
                 temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp",
                             torch.tensor(temperature).to(torch.device(f"cuda:{local_rank}")))
        self.register_buffer("neg_mask",
                             (~torch.eye(batch_size * 2,
                                         batch_size * 2,
                                         dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1),
                                  z.unsqueeze(0),
                                  dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class ssl_loss(torch.nn.Module):
    def __init__(self, batch_size, local_rank):
        super().__init__()
        # self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.device = torch.device(f"cuda:{local_rank}")
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(local_rank, batch_size).cuda()
        # self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self,
                 output_contrastive,
                 target_contrastive,
                 output_recons,
                 target_recons):
        # rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons.to(self.device))

        return contrast_loss, recon_loss

class SSL_Trainer(Trainer):
    r"""Trainer class for self-supervised learning.

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
                 checkpoint: Optional[str] = None):
        super(SSL_Trainer, self).__init__(cfg, device, mode, rank, checkpoint)
        self.loss = ssl_loss(cfg.SOLVER.SAMPLES_PER_BATCH, rank)

    def _train_misc_criterion_ssl(self, aug_1, aug_1_masked, aug_2, aug_2_masked, learner, iter_total):
        contrast_1, rec_1 = learner(aug_1_masked)
        contrast_2, rec_2 = learner(aug_2_masked)
        imgs_recon = torch.cat([rec_1, rec_2], dim=0)
        imgs = torch.cat([aug_1, aug_2], dim=0)
        loss_con, loss_rec = self.loss(contrast_1, contrast_2, imgs_recon, imgs)
        loss = loss_con + loss_rec
        self.backward_pass(loss)  # backward pass
        losses_vis = {'contrast': loss_con, 'inpainting': loss_rec}
        # logging and update record
        if hasattr(self, 'monitor'):
            do_vis = self.monitor.update(iter_total, loss, losses_vis,
                                         self.optimizer.param_groups[0]['lr'])
            if do_vis:
                # visualize representation distance next
                self.monitor.visualize(aug_1_masked, [rec_1], aug_2_masked, [[rec_2]], iter_total)
                if torch.cuda.is_available():
                    GPUtil.showUtilization(all=True)

        # Save model
        if (iter_total + 1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
            print("Save model checkpoint at iteration ", iter_total)
            state = {'iteration': iter_total + 1,
                     # Saving DataParallel or DistributedDataParallel models
                     'state_dict': learner.module.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict()}

            # Saves checkpoint to experiment directory
            filename = 'checkpoint_%05d.pth.tar' % (iter_total + 1)
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)
        # if (iter_total+1) % self.cfg.SOLVER.ITERATION_VAL == 0:
        #     self.validate(iter_total)

        # update learning rate
        self.maybe_update_swa_model(iter_total)
        self.scheduler_step(iter_total, loss)

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
        del aug_1, aug_2, loss, losses_vis

    def _train_misc_criterion_byol(self, aug_1, aug_2, learner, iter_total):
        loss_byol = learner(aug_1, aug_2)
        loss = loss_byol
        self.backward_pass(loss)  # backward pass
        self.model.module.update_moving_average()  # update moving average of target encoder
        losses_vis = {'byol': loss_byol}
        # logging and update record
        if hasattr(self, 'monitor'):
            do_vis = self.monitor.update(iter_total, loss, losses_vis,
                                         self.optimizer.param_groups[0]['lr'])
            do_vis = False
            if do_vis:
                # visualize representation distance next
                self.monitor.visualize(aug_1, aug_2, aug_1, aug_2, iter_total)
                if torch.cuda.is_available():
                    GPUtil.showUtilization(all=True)

        # Save model
        if (iter_total + 1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
            print("Save model checkpoint at iteration ", iter_total)
            state = {'iteration': iter_total + 1,
                     # Saving DataParallel or DistributedDataParallel models
                     'state_dict': learner.module.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict()}

            # Saves checkpoint to experiment directory
            filename = 'checkpoint_%05d.pth.tar' % (iter_total + 1)
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)
        # if (iter_total+1) % self.cfg.SOLVER.ITERATION_VAL == 0:
        #     self.validate(iter_total)

        # update learning rate
        self.maybe_update_swa_model(iter_total)
        self.scheduler_step(iter_total, loss)

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
        del aug_1, aug_2, loss, losses_vis



    def train(self):
        self.model.train()
        self.total_time = 0
        if self.cfg.MODEL.SSL == 'byol':
            for i in range(self.total_iter_nums):
                iter_total = self.start_iter + i
                self.start_time = time.perf_counter()
                self.optimizer.zero_grad()
                # load data
                sample = next(self.dataloader)
                volume = sample.out_input
                volume = torch.chunk(volume, 2, dim=2)
                aug_1 = torch.squeeze(volume[0], 2)
                aug_2 = torch.squeeze(volume[1], 2)
                self.data_time = time.perf_counter() - self.start_time
                self._train_misc_criterion_byol(aug_1, aug_2, self.model, iter_total)
        elif self.cfg.MODEL.SSL == 'rec+simclr':
            for i in range(self.total_iter_nums):
                iter_total = self.start_iter + i
                self.start_time = time.perf_counter()
                self.optimizer.zero_grad()
                # load data
                sample = next(self.dataloader)
                # volume = torch.transpose(sample.out_input, 3, 5)
                volume = sample.out_input
                volume = torch.chunk(volume, 2, dim=2)
                aug_1 = torch.squeeze(volume[0], 2)
                aug_2 = torch.squeeze(volume[1], 2)
                aug_1_masked = aug_rand(aug_1)
                aug_2_masked = aug_rand(aug_2)
                self.data_time = time.perf_counter() - self.start_time
                self._train_misc_criterion_ssl(aug_1, aug_1_masked, aug_2, aug_2_masked, self.model, iter_total)