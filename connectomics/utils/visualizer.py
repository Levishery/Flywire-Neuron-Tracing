from __future__ import print_function, division

import random
from typing import Optional, List, Union, Tuple

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
import PIL.Image as Image

from ..data.utils import decode_quantize
from connectomics.model.utils import SplitActivation

__all__ = [
    'Visualizer'
]


class Visualizer(object):
    """TensorboardX visualizer for displaying loss, learning rate and predictions
    at training time.
    """

    def __init__(self, cfg, vis_opt=0, N=16):
        self.cfg = cfg
        self.act = SplitActivation.build_from_cfg(cfg, do_cat=False)
        self.vis_opt = vis_opt
        self.N = N  # default maximum number of sections to show
        self.N_ind = None

        self.semantic_colors = {}
        for topt in self.cfg.MODEL.TARGET_OPT:
            if topt[0] == '9':
                channels = int(topt.split('-')[1])
                colors = [torch.rand(3) for _ in range(channels)]
                colors[0] = torch.zeros(3)  # make background black
                self.semantic_colors[topt] = torch.stack(colors, 0)

    def visualize(self, volume, label, output, weight, iter_total, writer,
                  suffix: Optional[str] = None):
        # split the prediction into chunks along the channel dimension
        volume = (volume * self.cfg.DATASET.STD) + self.cfg.DATASET.MEAN
        output = self.act(output)
        assert len(output) == len(label)

        for idx in range(len(self.cfg.MODEL.TARGET_OPT)):
            topt = self.cfg.MODEL.TARGET_OPT[idx]
            if topt[0] == '9':
                output[idx] = self.get_semantic_map(output[idx], topt)
                label[idx] = self.get_semantic_map(
                    label[idx], topt, argmax=False)
            if topt[0] == '5':
                output[idx] = decode_quantize(
                    output[idx], mode='max').unsqueeze(1)
                temp_label = label[idx].copy().astype(np.float32)[
                    :, np.newaxis]
                label[idx] = temp_label / temp_label.max() + 1e-6
            if topt[0] == 'e':
                output[idx] = self.emb2rgb(output[idx])
                if label[idx].dim() == 6:
                    label[idx] = label[idx][:, 1, :, :, :, :] / label[idx][:, 1, :, :, :, :].max() + 1e-6
                else:
                    label[idx] = label[idx] / label[idx].max() + 1e-6
            RGB = (topt[0] in ['1', '2', '9', 'e'])
            vis_name = self.cfg.MODEL.TARGET_OPT[idx] + '_' + str(idx)
            if suffix is not None:
                vis_name = vis_name + '_' + suffix
            if isinstance(label[idx], (np.ndarray, np.generic)):
                label[idx] = torch.from_numpy(label[idx])

            weight_maps = {}
            for j, wopt in enumerate(self.cfg.MODEL.WEIGHT_OPT[idx]):
                if wopt != '0':
                    w_name = vis_name + '_' + wopt
                    weight_maps[w_name] = weight[idx][j]

            self.visualize_consecutive(volume, label[idx], output[idx], weight_maps, iter_total,
                                       writer, RGB=RGB, vis_name=vis_name)

    def visualize_consecutive(self, volume, label, output, weight_maps, iteration,
                              writer, RGB=False, vis_name='0_0'):
        volume, label, output, weight_maps = self.prepare_data(
            volume, label, output, weight_maps)
        sz = volume.size()  # z,c,y,x
        label_sz = label.size()
        pad_y = int((sz[2]-label_sz[2])/2)
        pad_x = int((sz[3] - label_sz[3]) / 2)
        pad = (pad_x, pad_x, pad_y, pad_y)

        label = F.pad(label, pad, "constant", 0)
        output = F.pad(output, pad, "constant", 0)

        canvas = []
        volume_visual = volume.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
        canvas.append(volume_visual)

        def maybe2rgb(temp):
            if temp.shape[1] == 2: # 2d affinity map has two channels
                temp = torch.cat([temp, torch.zeros(
                    sz[0], 1, sz[2], sz[3]).type(temp.dtype)], dim=1)
            if temp.shape[1] == 1: # original label has one channels
                temp = torch.cat([temp, temp, temp], dim=1)
            return temp

        if RGB:
            output_visual = [maybe2rgb(output.detach().cpu())]
            label_visual = [maybe2rgb(label.detach().cpu())]
        else:
            output_visual = [self.vol_reshape(
                output[:, i], sz) for i in range(sz[1])]
            label_visual = [self.vol_reshape(
                label[:, i], sz) for i in range(sz[1])]

        weight_visual = []
        for key in weight_maps.keys():
            weight_maps[key] = F.pad(weight_maps[key], pad, "constant", 0)
            weight_visual.append(maybe2rgb(weight_maps[key]).detach().cpu().expand(
                                 sz[0], 3, sz[2], sz[3]))

        canvas = canvas + output_visual + label_visual + weight_visual
        canvas_merge = torch.cat(canvas, 0)
        canvas_show = vutils.make_grid(
            canvas_merge, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Consecutive_%s' % vis_name, canvas_show, iteration)

    def prepare_data(self, volume, label, output, weight_maps):
        ndim = len(volume.shape)
        assert ndim in [4, 5]
        is_3d = (ndim == 5)

        volume = self.permute_truncate(volume, is_3d)
        label = self.permute_truncate(label, is_3d)
        output = self.permute_truncate(output, is_3d)
        for key in weight_maps.keys():
            weight_maps[key] = self.permute_truncate(weight_maps[key], is_3d)

        return volume, label, output, weight_maps

    def permute_truncate(self, data, is_3d=False):
        if is_3d:
            data = data[0].permute(1, 0, 2, 3)
            start = int(data.size()[0] / 2 - self.N / 2)
        high = min(data.size()[0], self.N + start)
        return data[start:high]

    def get_semantic_map(self, output, topt, argmax=True):
        if isinstance(output, (np.ndarray, np.generic)):
            output = torch.from_numpy(output)
        # output shape: BCDHW or BCHW
        if argmax:
            output = torch.argmax(output, 1)
        pred = self.semantic_colors[topt][output]
        if len(pred.size()) == 4:   # 2D Inputs
            pred = pred.permute(0, 3, 1, 2)
        elif len(pred.size()) == 5:  # 3D Inputs
            pred = pred.permute(0, 4, 1, 2, 3)

        return pred

    def emb2rgb(self, x_emb):
        # x_emb = x_emb.squeeze(0)
        x_emb = np.array(x_emb.detach().cpu())
        shape = x_emb.shape  # b,e,d,h,w
        pca = PCA(n_components=3)
        x_emb = np.transpose(x_emb, [0, 2, 3, 4, 1])  # b,d,h,w,e
        x_emb = x_emb.reshape(-1, 16)  # b*d*h*w,e
        new_emb = pca.fit_transform(x_emb)  # b*d*h*w,3
        new_emb = new_emb.reshape(shape[0], shape[2], shape[3], shape[4], 3)  # b,d,h,w,3
        new_emb = np.transpose(new_emb, [0, 4, 1, 2, 3])
        new_emb = torch.tensor(new_emb)
        return new_emb

    def vol_reshape(self, vol, sz):
        vol = vol.detach().cpu().unsqueeze(1)
        return vol.expand(sz[0], 3, sz[2], sz[3])

    def plot_distance(self, distance_pos, distance_neg, iteration, writer, name=None):
        distance_neg = np.concatenate([np.asarray(sample.cpu()) for sample in distance_neg])
        distance_pos = np.asarray([np.asarray(sample.cpu()) for sample in distance_pos])
        random.shuffle(distance_neg)
        plt.scatter(np.random.rand(len(distance_neg)), np.asarray(distance_neg), c='black', s=5)
        plt.scatter(np.random.rand(len(distance_pos)), np.asarray(distance_pos), c='red', s=5)
        # from plt to np
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)
        plot_distance = image[:, :, :3]
        plot_distance.transpose(2,1,0)
        writer.add_image('%s distance scatter plot' % name, plot_distance, iteration, dataformats='HWC')
