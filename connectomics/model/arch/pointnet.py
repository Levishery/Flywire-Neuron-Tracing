import torch.nn as nn
import torch.nn.functional as F
from ..utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, MLPHead


class PointNetV2(nn.Module):
    def __init__(self, *args, **kwargs): # remove the num_class premeter
        super(PointNetV2, self).__init__()
        # in_channel = 3 if normal_channel else 0
        in_channel = 16+1
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.projetion = MLPHead(in_channels=1024, **kwargs['projection_head'])
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, features):
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024) # change 1024 to 2048
#         xxx = x                     #特征
#         #print("l3_points", l3_points.size())
#         #print(x)
#         #print("l3_points.view(B, 1024)", x.size())
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         #print("self.drop1(F.relu(self.bn1(self.fc1(x))))", x.size())
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         #print("self.drop2(F.relu(self.bn2(self.fc2(x))))", x.size())
#         x = self.fc3(x)
#         #print("self.fc3(x)", x.size())
#         #x = F.log_softmax(x, -1)
#         x = F.softmax(x, -1)
#         #print("F.log_softmax(x, -1)", x.size())


#         return x, xxx, l3_points #返回特征的时候取消注释
        return self.projetion(x), x, l3_points


# class get_loss(nn.Module):
#     def __init__(self, W = None):
#         super(get_loss, self).__init__()
#         self.W = W

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target, self.W)
#         #print(self.W)

#         return total_loss