# Code adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union
import torch
# from torch.optim.swa_utils import AveragedModel, SWALR
from yacs.config import CfgNode

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR
import itertools
import math
from copy import deepcopy
import warnings

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler

# 0. Gradient Clipping

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = cfg.clone()

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
        optimizer_type: Type[torch.optim.Optimizer], gradient_clipper: _GradientClipper
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                gradient_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer_type.__name__ + "WithGradientClip",
        (optimizer_type,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
        cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.Optimizer:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer instance of some type OptimizerType to become an instance
    of the new dynamically created class OptimizerTypeWithGradientClip
    that inherits OptimizerType and overrides the `step` method to
    include gradient clipping.
    Args:
        cfg: CfgNode
            configuration options
        optimizer: torch.optim.Optimizer
            existing optimizer instance
    Return:
        optimizer: torch.optim.Optimizer
            either the unmodified optimizer instance (if gradient clipping is
            disabled), or the same instance with adjusted __class__ to override
            the `step` method and include gradient clipping
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        type(optimizer), grad_clipper
    )
    optimizer.__class__ = OptimizerWithGradientClip
    return optimizer


# 1. Build Optimizer


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr,
                        "weight_decay": weight_decay}]

    name = cfg.SOLVER.NAME
    assert name in ["SGD", "Adam", "AdamW"]
    if name == "SGD":
        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, name)(
            params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS)
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    print('Optimizer: ', optimizer.__class__.__name__)
    return optimizer


# 2. Build LR Scheduler


def build_lr_scheduler(
        cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.ITERATION_TOTAL,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "MultiStepLR":
        return MultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA
        )
    elif name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min', factor=cfg.SOLVER.GAMMA, patience=1000,
            threshold=0.001, threshold_mode='rel', cooldown=0,
            min_lr=1e-06, eps=1e-08
        )
    elif name == 'OneCycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.SOLVER.BASE_LR,
            pct_start=cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER,
            final_div_factor=1000,
            total_steps=cfg.SOLVER.MAX_ITER
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))


# 3. Build SWA model


def build_swa_model(cfg: CfgNode,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    # Instead of copying weights during initialization, the SWA model copys
    # the model weights when self.update_parameters is first called.
    # https://github.com/pytorch/pytorch/blob/1.7/torch/optim/swa_utils.py#L107

    # The SWA model needs to be constructed for all processes in distributed
    # training, otherwise the training can get stuck.
    swa_model = AveragedModel(model)
    lr = cfg.SOLVER.BASE_LR
    lr *= cfg.SOLVER.SWA.LR_FACTOR
    swa_scheduler = SWALR(optimizer, swa_lr=lr)
    return swa_model, swa_scheduler


# torch.optim.swa_util
class AveragedModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter and the number of models already averaged; if None,
            equally weighted average is used (default: None)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)

    Example:
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    You can also use custom averaging functions with `avg_fn` parameter.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights.

    Example:
        >>> # Compute exponential moving averages of the weights and buffers
        >>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                            0.1 * averaged_model_parameter + 0.9 * model_parameter
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg, use_buffers=True)

    .. note::
        When using SWA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        This can be done either by using the :meth:`torch.optim.swa_utils.update_bn`
        or by setting :attr:`use_buffers` to `True`. The first approach updates the
        statistics in a post-training step by passing data through the model. The
        second does it during the parameter update phase by averaging all buffers.
        Empirical evidence has shown that updating the statistics in normalization
        layers increases accuracy, but you may wish to empirically test which
        approach yields the best results in your problem.

    .. note::
        :attr:`avg_fn` is not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    """

    def __init__(self, model, device=None, avg_fn=None, use_buffers=False):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                       (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1


@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class SWALR(_LRScheduler):
    r"""Anneals the learning rate in each parameter group to a fixed value.

    This learning rate scheduler is meant to be used with Stochastic Weight
    Averaging (SWA) method (see `torch.optim.swa_utils.AveragedModel`).

    Args:
        optimizer (torch.optim.Optimizer): wrapped optimizer
        swa_lrs (float or list): the learning rate value for all param groups
            together or separately for each group.
        annealing_epochs (int): number of epochs in the annealing phase
            (default: 10)
        annealing_strategy (str): "cos" or "linear"; specifies the annealing
            strategy: "cos" for cosine annealing, "linear" for linear annealing
            (default: "cos")
        last_epoch (int): the index of the last epoch (default: -1)

    The :class:`SWALR` scheduler is can be used together with other
    schedulers to switch to a constant learning rate late in the training
    as in the example below.

    Example:
        >>> loader, optimizer, model = ...
        >>> lr_lambda = lambda epoch: 0.9
        >>> scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
        >>>        lr_lambda=lr_lambda)
        >>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
        >>>        anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05)
        >>> swa_start = 160
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    """

    def __init__(self, optimizer, swa_lr, anneal_epochs=10, anneal_strategy='cos', last_epoch=-1):
        swa_lrs = self._format_param(optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups):
            group['swa_lr'] = swa_lr
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', "
                             f"instead got {anneal_strategy}")
        elif anneal_strategy == 'cos':
            self.anneal_func = self._cosine_anneal
        elif anneal_strategy == 'linear':
            self.anneal_func = self._linear_anneal
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}")
        self.anneal_epochs = anneal_epochs
        super(SWALR, self).__init__(optimizer, last_epoch)

    @staticmethod
    def _format_param(optimizer, swa_lrs):
        if isinstance(swa_lrs, (list, tuple)):
            if len(swa_lrs) != len(optimizer.param_groups):
                raise ValueError("swa_lr must have the same length as "
                                 f"optimizer.param_groups: swa_lr has {len(swa_lrs)}, "
                                 f"optimizer.param_groups has {len(optimizer.param_groups)}")
            return swa_lrs
        else:
            return [swa_lrs] * len(optimizer.param_groups)

    @staticmethod
    def _linear_anneal(t):
        return t

    @staticmethod
    def _cosine_anneal(t):
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        step = self._step_count - 1
        if self.anneal_epochs == 0:
            step = max(1, step)
        prev_t = max(0, min(1, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [self._get_initial_lr(group['lr'], group['swa_lr'], prev_alpha)
                    for group in self.optimizer.param_groups]
        t = max(0, min(1, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        return [group['swa_lr'] * alpha + lr * (1 - alpha)
                for group, lr in zip(self.optimizer.param_groups, prev_lrs)]
