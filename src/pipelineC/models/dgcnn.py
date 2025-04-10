import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    Finds k nearest neighbors for each point in the point cloud
    
    Args:
        x: Point cloud data (B, C, N)
        k: Number of nearest neighbors
    
    Returns:
        idx: Indices of k-nearest neighbors (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Constructs edge features for each point by subtracting the feature of each point from its neighbors
    
    Args:
        x: Point cloud features (B, C, N)
        k: Number of nearest neighbors
        idx: Precomputed indices of k-nearest neighbors
        
    Returns:
        Edge features (B, 2*C, N, k)
    """
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)
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

class DGCNN_Seg(nn.Module):
    """DGCNN model for point cloud segmentation"""
    
    def __init__(self, k=20, num_classes=2, emb_dims=1024, dropout=0.5):
        """
        Initialize DGCNN segmentation model
        
        Args:
            k: Number of nearest neighbors
            num_classes: Number of segmentation classes (2 for table/non-table)
            emb_dims: Dimension of embeddings
            dropout: Dropout probability
        """
        super(DGCNN_Seg, self).__init__()
        self.k = k
        
        # EdgeConv Layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # MLP for per-point segmentation
        self.convs1 = nn.Conv1d(256, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, num_classes, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input point cloud (B, N, 3)
            
        Returns:
            Point-wise segmentation logits (B, N, num_classes)
        """
        batch_size, num_points, _ = x.size()
        x = x.transpose(2, 1) # (B, 3, N)
        
        # EdgeConv Block 1
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        
        # EdgeConv Block 2
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        # EdgeConv Block 3
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        # EdgeConv Block 4
        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        
        # Concatenate features from all blocks
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 64*4, N)
        
        # Per-point prediction
        x = F.leaky_relu(self.bns1(self.convs1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bns2(self.convs2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.convs3(x)
        
        x = x.transpose(2, 1).contiguous()  # (B, N, num_classes)
        
        return x