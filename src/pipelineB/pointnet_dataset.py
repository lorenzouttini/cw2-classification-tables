import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudTableDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, normalize=True):
        """
        root_dir: path to folder with .npy files
        num_points: number of points to sample from each cloud
        normalize: if True, centers cloud at origin
        """
        self.files = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ])
        self.num_points = num_points
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        points = data["points"]
        label = data["label"]

        # Filter out NaNs or Infs
        mask = np.all(np.isfinite(points), axis=1)
        points = points[mask]

        # Random sampling
        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]

        if self.normalize:
            points = points - np.mean(points, axis=0)

        return torch.tensor(points, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
