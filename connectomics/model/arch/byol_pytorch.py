import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .swin_unetr import get_downsample
from torchvision import transforms as T

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):

        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size_xy,
        image_size_z,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        use_momentum = True
    ):
        super().__init__()
        # default SimCLR augmentation
        # the augmentations is moved to dataset_volume.py

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        self.device = get_module_device(net)
        self.to(self.device)

        # send a (batch size of 2 for batchnorm in the projection layer) mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 1, image_size_z, image_size_xy, image_size_xy).cuda(), torch.randn(2, 1, image_size_z, image_size_xy, image_size_xy).cuda())

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        image_one, image_two,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and image_one.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
        if return_embedding:
            return self.online_encoder(image_one, return_projection = return_projection)
        # sample = {}
        # sample['image'] = x
        # sample['label'] = x
        # image_one, image_two = self.augment1(x).to(self.device), self.augment2(x).to(self.device)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()


class SSL(nn.Module):
    def __init__(
        self,
        net,
    ):
        super().__init__()
        # default SimCLR augmentation
        # the augmentations is moved to dataset_volume.py
        dim = 384
        self.swinViT = net
        self.isotropy = net.isotropy
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        self.conv = nn.Sequential(nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm3d(dim // 2),
                                  nn.LeakyReLU(),
                                  nn.Upsample(scale_factor=tuple(get_downsample(is_isotropic=self.isotropy[4])), mode='trilinear', align_corners=False),
                                  nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm3d(dim // 4),
                                  nn.LeakyReLU(),
                                  nn.Upsample(scale_factor=tuple(get_downsample(is_isotropic=self.isotropy[3])), mode='trilinear', align_corners=False),
                                  nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm3d(dim // 8),
                                  nn.LeakyReLU(),
                                  nn.Upsample(scale_factor=tuple(get_downsample(is_isotropic=self.isotropy[2])), mode='trilinear', align_corners=False),
                                  nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm3d(dim // 16),
                                  nn.LeakyReLU(),
                                  nn.Upsample(scale_factor=tuple(get_downsample(is_isotropic=self.isotropy[1])), mode='trilinear', align_corners=False),
                                  nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm3d(dim // 16),
                                  nn.LeakyReLU(),
                                  nn.Upsample(scale_factor=tuple(net.patch_size), mode='trilinear', align_corners=False),
                                  nn.Conv3d(dim // 16, 1, kernel_size=1, stride=1)
                                  )
        # get device of network and make wrapper same device
        self.device = get_module_device(net)
        self.to(self.device)

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        _, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        # x_rot = self.rotation_pre(x4_reshape[:, 0])
        # x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)

        return x_contrastive, x_rec
