from __future__ import print_function, division
from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """DICE loss.
    """
    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    def __init__(self, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.

        for index in range(pred.size()[0]):
            iflat = pred[index].contiguous().view(-1)
            tflat = target[index].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            if self.power == 1:
                loss += 1 - ((2. * intersection + self.smooth) /
                             (iflat.sum() + tflat.sum() + self.smooth))
            else:
                loss += 1 - ((2. * intersection + self.smooth) /
                             ((iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        if self.power == 1:
            loss = 1 - ((2. * intersection + self.smooth) /
                        (iflat.sum() + tflat.sum() + self.smooth))
        else:
            loss = 1 - ((2. * intersection + self.smooth) /
                        ((iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))
        return loss

    def forward(self, pred, target, weight_mask=None):
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(
                target.size(), pred.size()))

        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:
            loss = self.dice_loss_batch(pred, target)
        return loss


class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, pred, target, weight=None):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).to(pred.device)
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight_mask=None):
        return self.weighted_mse_loss(pred, target, weight_mask)


class WeightedMAE(nn.Module):
    """Mask weighted mean absolute error (MAE) energy function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss * weight_mask
        return loss.mean()


class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """

    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight_mask=None):
        return F.binary_cross_entropy(pred, target, weight_mask)


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted binary cross-entropy with logits.
    """

    def __init__(self, size_average=True, reduce=True, eps=0.):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.eps = eps

    def forward(self, pred, target, weight_mask=None):
        return F.binary_cross_entropy_with_logits(pred, target.clamp(self.eps,1-self.eps), weight_mask)


class WeightedCE(nn.Module):
    """Mask weighted multi-class cross-entropy (CE) loss.
    """

    def __init__(self, class_weight: Optional[List[float]] = None):
        super().__init__()
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight)

    def forward(self, pred, target, weight_mask=None):
        # Different from, F.binary_cross_entropy, the "weight" parameter
        # in F.cross_entropy is a manual rescaling weight given to each
        # class. Therefore we need to multiply the weight mask after the
        # loss calculation.
        if self.class_weight is not None:
            self.class_weight = self.class_weight.to(pred.device)

        loss = F.cross_entropy(
            pred, target, weight=self.class_weight, reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


class WeightedLS(nn.Module):
    """Weighted CE loss with label smoothing (LS). The code is based on:
    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    """
    dim = 1

    def __init__(self, classes=10, cls_weights=None, smoothing=0.2):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

        self.weights = 1.0
        if cls_weights is not None:
            self.weights = torch.tensor(cls_weights)

    def forward(self, pred, target, weight_mask=None):
        shape = (1, -1, 1, 1, 1) if pred.ndim == 5 else (1, -1, 1, 1)
        if isinstance(self.weights, torch.Tensor) and self.weights.ndim == 1:
            self.weights = self.weights.view(shape).to(pred.device)

        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist*pred*self.weights, dim=self.dim)
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()

class WeightedBCEFocalLoss(nn.Module):
    """Weighted binary focal loss with logits.
    """
    def __init__(self, gamma=2., alpha=0.25, eps=0.):
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, weight_mask=None):
        pred_sig = pred.sigmoid()
        pt = (1-target)*(1-pred_sig) + target * pred_sig
        at = (1-self.alpha) * target + self.alpha * (1-target)
        wt = at * (1 - pt)**self.gamma
        if weight_mask is not None:
            wt *= weight_mask
        # return -(wt * pt.log()).mean() # log causes overflow
        bce = F.binary_cross_entropy_with_logits(pred, target.clamp(self.eps,1-self.eps), reduction='none')
        return (wt *  bce).mean()


class WSDiceLoss(nn.Module):
    def __init__(self, smooth=100.0, power=2.0, v2=0.85, v1=0.15):
        super().__init__()
        self.smooth = smooth
        self.power = power
        self.v2 = v2
        self.v1 = v1

    def dice_loss(self, pred, target):
        iflat = pred.reshape(pred.shape[0], -1)
        tflat = target.reshape(pred.shape[0], -1)
        wt = tflat * (self.v2 - self.v1) + self.v1
        g_pred = wt*(2*iflat - 1)
        g = wt*(2*tflat - 1)
        intersection = (g_pred * g).sum(-1)
        loss = 1 - ((2. * intersection + self.smooth) /
                    ((g_pred**self.power).sum(-1) + (g**self.power).sum(-1) + self.smooth))
        # loss = -torch.log10((2. * intersection + self.smooth) /
        #             ((g_pred**self.power).sum(-1) + (g**self.power).sum(-1) + self.smooth))

        return loss.mean()

    def forward(self, pred, target, weight_mask=None):
        loss = self.dice_loss(pred, target)
        return loss


class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v=0.5, delta_d=3, alpha=1, beta=1, gama=0.001):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gama = gama

    def discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]
        embed_dim = embedding.shape[1]
        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_id = len(labels)
            if num_id == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for idx in labels:
                seg_mask_i = (seg_gt_b == idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]  # get positive positions
                # print(embedding_i.shape)
                mean_i = torch.mean(embedding_i, dim=1)  # ????
                # print(mean_i.shape)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------

                var_loss = var_loss + torch.mean(
                    F.relu(torch.norm(embedding_i - mean_i.reshape(embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_id
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_id > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)  # shape (num_id, num_id)
                dist = dist + torch.eye(num_id, dtype=dist.dtype,
                                        device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d) ** 2) / (num_id * (num_id - 1)) / 2

            # reg_loss is not used in original paper
            reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size

        Loss = self.alpha * var_loss + self.beta * dist_loss + self.gama * reg_loss
        return Loss

    def forward(self, pred, target):
        loss = self.discriminative_loss(pred, target)
        return loss