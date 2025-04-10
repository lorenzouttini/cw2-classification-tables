import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Helper functions for DGCNN
# ------------------------

def knn(x, k):
    """
    x: (B, C, N) tensor
    Returns: indices of the k nearest neighbors for each point, (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)         # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]          # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    x: (B, C, N) tensor
    Returns: (B, 2*C, N, k) tensor representing edge features 
    """
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

# ------------------------
# DGCNN Model Definition
# ------------------------

class DGCNNClassifier(nn.Module):
    def __init__(self, k=20, emb_dims=1024, num_classes=2, dropout=0.5):
        super(DGCNNClassifier, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, 3)
        batch_size = x.size(0)
        x = x.transpose(2, 1)  # (B, 3, N)
        x = get_graph_feature(x, k=self.k)  # (B, 6, N, k)
        x = self.conv1(x)                   # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)
        
        x = get_graph_feature(x1, k=self.k)  # (B, 128, N, k)
        x = self.conv2(x)                   # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)
        
        x = get_graph_feature(x2, k=self.k)  # (B, 128, N, k)
        x = self.conv3(x)                   # (B, 128, N, k)
        x3 = x.max(dim=-1, keepdim=False)[0] # (B, 128, N)
        
        x = get_graph_feature(x3, k=self.k)  # (B, 256, N, k)
        x = self.conv4(x)                   # (B, 256, N, k)
        x4 = x.max(dim=-1, keepdim=False)[0] # (B, 256, N)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        x = self.conv5(x)                     # (B, emb_dims, N)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)            # (B, emb_dims*2)
        
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
