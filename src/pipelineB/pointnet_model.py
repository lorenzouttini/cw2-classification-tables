import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, D, N] or None
        # This version does not do real sampling/grouping (for simplicity)
        B, _, N = xyz.shape
        new_xyz = xyz[:, :, :self.npoint]  # naive subsampling
        new_points = points[:, :, :self.npoint] if points is not None else new_xyz

        new_points = new_points.unsqueeze(3)  # [B, D, N, 1]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]  # global max pooling
        return new_xyz, new_points

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256])

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, N, 3]
        x = x.permute(0, 2, 1)  # [B, 3, N]
        xyz, points = self.sa1(x, None)
        _, points = self.sa2(xyz, points)
        x = torch.max(points, 2)[0]  # [B, D]
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.fc2(x)
        return x