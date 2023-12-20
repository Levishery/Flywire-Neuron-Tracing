import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, feature_channel=None):
        super(get_model, self).__init__()
        self.feature_channel = feature_channel
        in_channel = 3+feature_channel if feature_channel else 3
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.feature_channel:
            feature = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            feature = None
        l1_xyz, l1_points = self.sa1(xyz, feature)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = self.Sigmoid(x)


        return x, l3_points



class get_loss(nn.Module):
    # focal loss
    def __init__(self, alpha=0.5):
        super(get_loss, self).__init__()
        self.loss = nn.BCELoss()
        self.alpha = alpha
    def forward(self, pred, target, trans_feat):
        if self.alpha > 0:
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = F.binary_cross_entropy(pred.squeeze(), target, reduction='none') * alpha_weight
            total_loss = torch.mean(loss)
        else:
            total_loss = self.loss(pred.squeeze(), target)

        return total_loss
