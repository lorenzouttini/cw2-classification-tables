import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import cv2
import pickle


def load_intrinsics(intrinsics_path):
    try:
        # Try to read intrinsics file
        with open(intrinsics_path, 'r') as f:
            content = f.read()
            
        # Parse intrinsics from file
        if "Depth Camera Intrinsics" in content:
            # Try to extract depth camera values
            try:
                fx = float(content.split("FX: ")[1].split(",")[0])
                fy = float(content.split("FY: ")[1].split("\n")[0])
                ppx = float(content.split("PPX: ")[1].split(",")[0])
                ppy = float(content.split("PPY: ")[1].split("\n")[0])
                
                return np.array([
                    [fx,  0.0, ppx],
                    [0.0, fy,  ppy],
                    [0.0, 0.0, 1.0]
                ])
            except:
                print(f"Warning: Could not parse depth camera intrinsics from {intrinsics_path}")
        
        # If we got here, either the file format is different or parsing failed
        # Try a more general approach to parse any 3x3 matrix
        lines = content.strip().split('\n')
        matrix_lines = [line for line in lines if line.strip() and line[0].isdigit() or line[0] == '-']
        
        if len(matrix_lines) >= 3:
            matrix = []
            for i in range(3):
                try:
                    row = [float(val) for val in matrix_lines[i].split()]
                    if len(row) >= 3:
                        matrix.append(row[:3])  # Just take first 3 values
                except:
                    pass
            
            if len(matrix) == 3 and all(len(row) == 3 for row in matrix):
                return np.array(matrix)
    
    except Exception as e:
        print(f"Warning: Error loading intrinsics from {intrinsics_path}: {e}")
    
    # Default values based on the RealSense depth camera
    print(f"Using default RealSense depth camera intrinsics")
    return np.array([
        [390.74, 0.0,   320.09],
        [0.0,   390.74, 244.11],
        [0.0,   0.0,    1.0]
    ])

def depth_to_pointcloud(depth_img, intrinsics):
    # Check if the image has more than 2 dimensions
    if len(depth_img.shape) > 2:
        # print("Depth image has more than 2 dimensions: ", depth_img.shape)

        # If it's a color image or has channels, convert to grayscale or take first channel
        if len(depth_img.shape) == 3 and depth_img.shape[2] == 3:
            # Convert RGB to grayscale
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_RGB2GRAY)
        else:
            # Just take the first channel
            depth_img = depth_img[:, :, 0]
    
    # Now we can safely unpack
    height, width = depth_img.shape
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_img.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pointcloud = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    valid = (Z.reshape(-1) > 0)
    pointcloud = pointcloud[valid]
    return pointcloud

def downsample_pointcloud(pointcloud, num_points=1024):
    N = pointcloud.shape[0]
    if N >= num_points:
        # Take equally spaced points (more evenly distributed than random)
        indices = np.linspace(0, N-1, num_points, dtype=int)
    else:
        # Repeat points in a deterministic way
        indices = np.arange(N).repeat(num_points // N + 1)[:num_points]
    return pointcloud[indices]

def random_augmentation(sample):
    # Augmentation function; can be set to None if not used.
    pointcloud = sample["pointcloud"].numpy()
    angle = np.random.uniform(0, 2*np.pi)
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0,             0,              1]])
    pointcloud = pointcloud @ R.T
    scale = np.random.uniform(0.9, 1.1)
    pointcloud *= scale
    jitter = np.random.normal(0, 0.01, pointcloud.shape)
    pointcloud += jitter
    sample["pointcloud"] = torch.from_numpy(pointcloud).float()
    return sample

class TableClassificationDataset(Dataset):
    def __init__(self, root_dir, depth_folder="depth", annotation_path="tabletop_labels.dat",
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

        if os.path.exists(self.annotation_path):
            with open(self.annotation_path, 'rb') as f:
                self.annotations = pickle.load(f)
            if self.verbose:
                print(f"Loaded annotations from {self.annotation_path} with {len(self.annotations)} entries.")
                
        else:
            print(f"Annotation file not found at {self.annotation_path}. Using default empty annotations.")
            self.annotations = [[] for _ in range(len(self.depth_files))]

        self.intrinsics = load_intrinsics(intrinsics_path)
        if len(self.annotations) != len(self.depth_files):
            print("Warning: Number of depth images does not match number of annotation entries!")
    
    def __len__(self):
        return len(self.depth_files)
    
    def __getitem__(self, idx):
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        # print("Depth path: ", depth_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        depth_img = depth_img.astype(np.float32)

        pointcloud = depth_to_pointcloud(depth_img, self.intrinsics)
        pointcloud = downsample_pointcloud(pointcloud, self.num_points)

        annotation = self.annotations[idx]
        label = 1 if len(annotation) > 0 else 0
        
        pointcloud = torch.from_numpy(pointcloud).float()
        label = torch.tensor(label, dtype=torch.long)
        sample = {
            "pointcloud": pointcloud,
            "label": label,
            "file_path": depth_path   # added key to remember the original image path
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

def process_sequences(seq_list, base_path, set_name="Set"):
    ds_list = []
    for seq in seq_list:
        seq_path = os.path.join(base_path, seq)
        
        # Define the depth folder and annotation path
        depth_folder = "depth"
        annotation_path = "labels/tabletop_labels.dat"
        intrinsics_path = os.path.join(seq_path, "camera_intrinsics.txt")
        dataset = TableClassificationDataset(
            root_dir=seq_path,
            depth_folder=depth_folder,
            annotation_path=annotation_path,
            intrinsics_path=intrinsics_path,
            num_points=1024,
            transform=None,  # No transform
            verbose=False
        )
        pos_count = sum(1 for i in range(len(dataset)) if dataset[i]["label"].item() == 1)
        neg_count = len(dataset) - pos_count
        print(f"{set_name} '{seq}' has {len(dataset)} PNG depth images: {pos_count} positives, {neg_count} negatives.")
        ds_list.append(dataset)
    
    combined = ConcatDataset(ds_list)
    
    total_pos = sum(1 for i in range(len(combined)) if combined[i]["label"].item() == 1)
    total_neg = len(combined) - total_pos
    print(f"{set_name} Combined dataset has {len(combined)} images: {total_pos} positives, {total_neg} negatives.")
    return combined

# Helper class to apply transformations to a dataset
class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == "__main__":
    # Define test sequences.
    ucl_data_sequences = [
        "UCL_Data"
    ]

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two directories to reach project root
    base_path = os.path.normpath(os.path.join(script_dir, "../../data/RealSense/UCL_Data/"))
    
    print("Processing Training Sequences:")
    test_dataset = process_sequences(ucl_data_sequences, base_path, set_name="Test")