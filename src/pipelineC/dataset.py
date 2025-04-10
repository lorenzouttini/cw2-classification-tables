import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import cv2
import pickle
from matplotlib.path import Path

def load_intrinsics(intrinsics_path=None):
    """Load camera intrinsics from file or use default values."""
    if intrinsics_path and os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
        K = []
        for line in lines:
            nums = list(map(float, line.strip().split()))
            K.append(nums)
        K = np.array(K)
    else:
        K = np.array([[570.3422, 0, 320],
                      [0, 570.3422, 240],
                      [0, 0, 1]])
    return K

def depth_to_pointcloud(depth_img, intrinsics):
    """Convert depth image to 3D point cloud."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    height, width = depth_img.shape
    
    # Create meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert depth to float
    Z = depth_img.astype(np.float32)
    
    # Back-project to 3D
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack to form point cloud
    pointcloud = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    
    # Keep only valid points (non-zero depth)
    valid = (Z.reshape(-1) > 0)
    pointcloud = pointcloud[valid]
    
    # Keep track of pixel coordinates for labeling
    pixel_coords = np.stack((u, v), axis=-1).reshape(-1, 2)
    pixel_coords = pixel_coords[valid]
    
    return pointcloud, pixel_coords

def class_balanced_downsample(pointcloud, point_labels, num_points=1024, table_ratio=0.7):
    """Downsample point cloud with emphasis on preserving table points."""
    # Get indices for each class
    table_idx = np.where(point_labels == 1)[0]
    background_idx = np.where(point_labels == 0)[0]
    
    num_table_points = len(table_idx)
    num_background_points = len(background_idx)
    
    # Calculate target counts for each class
    target_table_points = min(int(num_points * table_ratio), num_table_points)
    target_bg_points = num_points - target_table_points
    
    # If we don't have enough points of either class, adjust ratios
    if target_bg_points > num_background_points:
        target_bg_points = num_background_points
        target_table_points = min(num_points - target_bg_points, num_table_points)
    
    # Sample points from each class
    if num_table_points > 0 and target_table_points > 0:
        sampled_table_idx = np.random.choice(table_idx, target_table_points, 
                                            replace=(target_table_points > num_table_points))
    else:
        sampled_table_idx = np.array([], dtype=int)
        
    if num_background_points > 0 and target_bg_points > 0:
        sampled_bg_idx = np.random.choice(background_idx, target_bg_points, 
                                         replace=(target_bg_points > num_background_points))
    else:
        sampled_bg_idx = np.array([], dtype=int)
    
    # Combine indices and gather points/labels
    sampled_idx = np.concatenate([sampled_table_idx, sampled_bg_idx])
    
    # Edge case: if we have fewer points than requested
    if len(sampled_idx) < num_points:
        additional_idx = np.random.choice(sampled_idx, num_points - len(sampled_idx), replace=True)
        sampled_idx = np.concatenate([sampled_idx, additional_idx])
    
    # Shuffle the indices to avoid having all table points first
    np.random.shuffle(sampled_idx)
    
    downsampled_pointcloud = pointcloud[sampled_idx]
    downsampled_labels = point_labels[sampled_idx]
    
    return downsampled_pointcloud, downsampled_labels

class PointCloudSegmentationDataset(Dataset):
    """Point Cloud Segmentation Dataset for tabletop scenes."""
    def __init__(self, root_dir, depth_folder="depthTSDF", annotation_path="labels/tabletop_labels.dat",
                 intrinsics_path=None, num_points=1024, transform=None, verbose=False):
        super().__init__()
        self.root_dir = root_dir
        self.depth_dir = os.path.join(root_dir, depth_folder)
        self.annotation_path = os.path.join(root_dir, annotation_path)
        self.num_points = num_points
        self.transform = transform
        self.verbose = verbose

        # Only count PNG images
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.lower().endswith('.png')])
        if self.verbose:
            print(f"Found {len(self.depth_files)} PNG depth images in {self.depth_dir}")

        # Define blacklist for problematic images
        BLACKLIST = {
            "mit_76_studyroom": ["0002122-000070776179.png"],
            "mit_32_d507": ["0004668-000155734290.png"],
            "harvard_c11": ["0000006-000000166846.png"],
            "mit_lab_hj": [
                "0000281-000011389077.png",
                "0000676-000027372948.png",
                "0001107-000044791697.png",
                "0001327-000053667917.png"
            ]
        }
        
        # Track blacklisted indices for annotation adjustment
        self.blacklisted_indices = []
        
        # Extract dataset name from root_dir path to match with blacklist keys
        dataset_name = None
        for key in BLACKLIST.keys():
            if key in root_dir:
                dataset_name = key
                break
        
        # Filter out blacklisted files if this dataset has any
        if dataset_name and dataset_name in BLACKLIST:
            orig_count = len(self.depth_files)
            # Keep track of which indices we're removing
            for i, filename in enumerate(self.depth_files):
                if filename in BLACKLIST[dataset_name]:
                    self.blacklisted_indices.append(i)
            
            # Filter the depth files list
            self.depth_files = [f for f in self.depth_files if f not in BLACKLIST[dataset_name]]
            filtered_count = orig_count - len(self.depth_files)
            
            if self.verbose and filtered_count > 0:
                print(f"Filtered out {filtered_count} blacklisted images from {dataset_name}")
        
        # Load annotations
        if os.path.exists(self.annotation_path):
            with open(self.annotation_path, 'rb') as f:
                self.annotations = pickle.load(f)
            if self.verbose:
                print(f"Loaded annotations from {self.annotation_path} with {len(self.annotations)} entries.")
                
            # Remove annotations for blacklisted files
            if self.blacklisted_indices:
                self.annotations = [anno for i, anno in enumerate(self.annotations) if i not in self.blacklisted_indices]
                if self.verbose:
                    print(f"Adjusted annotations list to match filtered files")
        else:
            if self.verbose:
                print(f"Annotation file not found at {self.annotation_path}. Using default empty annotations.")
            self.annotations = [[] for _ in range(len(self.depth_files))]

        self.intrinsics = load_intrinsics(intrinsics_path)
        if len(self.annotations) != len(self.depth_files) and self.verbose:
            print(f"Warning: Number of depth images ({len(self.depth_files)}) does not match " +
                  f"number of annotation entries ({len(self.annotations)})!")

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        depth_img = depth_img.astype(np.float32)

        # Get annotations for this image (list of polygons)
        polygons = self.annotations[idx]
        has_table = len(polygons) > 0
        
        # Convert depth image to point cloud with pixel coordinates
        pointcloud, pixel_coords = depth_to_pointcloud(depth_img, self.intrinsics)
        
        # Generate point-wise labels (0 for background, 1 for table)
        point_labels = np.zeros(len(pointcloud), dtype=np.int64)
        
        if has_table:
            for polygon in polygons:
                # Check if points are inside the polygon
                path = Path(list(zip(polygon[0], polygon[1])))
                mask = path.contains_points(pixel_coords)
                point_labels[mask] = 1
        
        # Downsample point cloud and labels
        if len(pointcloud) > 0:
            pointcloud, point_labels = class_balanced_downsample(
                pointcloud, point_labels, self.num_points, table_ratio=0.7)
        else:
            # Handle empty point clouds
            pointcloud = np.zeros((self.num_points, 3), dtype=np.float32)
            point_labels = np.zeros(self.num_points, dtype=np.int64)

        # Convert to torch tensors
        pointcloud = torch.from_numpy(pointcloud).float()
        point_labels = torch.from_numpy(point_labels).long()
        scene_label = torch.tensor(1 if has_table else 0, dtype=torch.long)
        
        sample = {
            "pointcloud": pointcloud, 
            "point_labels": point_labels,
            "scene_label": scene_label,
            "filename": self.depth_files[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def random_augmentation(sample):
    """Apply random augmentation to point cloud data."""
    pointcloud = sample["pointcloud"].numpy()
    point_labels = sample["point_labels"]  # Preserve labels during augmentation
    
    # Random rotation around z-axis
    angle = np.random.uniform(0, 2*np.pi)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    pointcloud = pointcloud @ R.T
    
    # Random scaling
    scale = np.random.uniform(0.9, 1.1)
    pointcloud *= scale
    
    # Random jitter
    jitter = np.random.normal(0, 0.01, pointcloud.shape)
    pointcloud += jitter
    
    sample["pointcloud"] = torch.from_numpy(pointcloud).float()
    return sample

def create_datasets(base_path, sequences_train, sequences_test, transform=None):
    """Create training and testing datasets from sequences."""
    train_datasets = []
    test_datasets = []
    
    # Process training sequences
    print("Processing training datasets:")
    for seq in sequences_train:
        seq_path = os.path.join(base_path, seq)
        # Determine the depth folder
        if "harvard_tea_2" in seq:
            depth_folder = "depth"
        else:
            depth_folder = "depthTSDF"
            
        dataset = PointCloudSegmentationDataset(
            root_dir=seq_path,
            depth_folder=depth_folder,
            annotation_path="labels/tabletop_labels.dat",
            intrinsics_path=os.path.join(seq_path, "intrinsics.txt"),
            num_points=1024,
            transform=transform,
            verbose=True
        )
        
        train_datasets.append(dataset)
    
    # Process test sequences
    print("\nProcessing test datasets:")
    for seq in sequences_test:
        seq_path = os.path.join(base_path, seq)
        # Determine the depth folder
        if "harvard_tea_2" in seq:
            depth_folder = "depth"
        else:
            depth_folder = "depthTSDF"
            
        dataset = PointCloudSegmentationDataset(
            root_dir=seq_path,
            depth_folder=depth_folder,
            annotation_path="labels/tabletop_labels.dat",
            intrinsics_path=os.path.join(seq_path, "intrinsics.txt"),
            num_points=1024,
            transform=None,  # No augmentation for test
            verbose=True
        )
        
        test_datasets.append(dataset)
       
    # Combine datasets
    train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    test_dataset = ConcatDataset(test_datasets) if test_datasets else None
    
    return train_dataset, test_dataset

def create_train_val_split(train_dataset, val_ratio=0.2, seed=42):
    """Split a dataset into training and validation sets"""
    from torch.utils.data import random_split
    
    # Calculate split sizes
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    
    # Split the dataset
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)  # For reproducibility
    )
    
    print(f"Split training data: {train_size} train samples, {val_size} validation samples")
    
    return train_subset, val_subset