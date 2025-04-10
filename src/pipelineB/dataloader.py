import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import cv2
import pickle
from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D visualization

# Blacklist for sequences
BLACKLIST = {
    "mit_76_studyroom/76-1studyroom2": ['0002122-000070776179.png'],
    "mit_32_d507/d507_2": ['0004668-000155734290.png'],
    "harvard_c11/hv_c11_2": ['0000006-000000166846.png'],
    "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika": [
        '0000281-000011389077.png',
        '0000676-000027372948.png',
        '0001107-000044791697.png',
        '0001327-000053667917.png'
    ]
}

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def load_intrinsics(intrinsics_path=None):
    """Load intrinsics file if it exists, otherwise use default values."""
    if intrinsics_path and os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
        K = []
        for line in lines:
            nums = list(map(float, line.strip().split()))
            K.append(nums)
        K = np.array(K)
        print(f"Loaded intrinsics from {intrinsics_path}:")
        print(K)
    else:
        print("Using default intrinsics. Cannot load from file:", intrinsics_path)
        K = np.array([[570.3422, 0, 320],
                      [0, 570.3422, 240],
                      [0, 0, 1]])
        print(K)
    return K

def depth_to_pointcloud(depth_img, intrinsics):
    """
    Calculate 3D point cloud (X, Y, Z) and 2D pixel coordinates for each point from depth image.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    height, width = depth_img.shape

    # Create meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_img.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Point cloud: (X, Y, Z) coordinates for each pixel
    pointcloud = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # Use only pixels with depth value greater than 0
    valid = (Z.reshape(-1) > 0)
    pointcloud = pointcloud[valid]

    # Calculate corresponding pixel coordinates
    pixel_coords = np.stack((u, v), axis=-1).reshape(-1, 2)
    pixel_coords = pixel_coords[valid]

    return pointcloud, pixel_coords

# def is_point_in_polygon(point, polygon):
#     """
#     Checks if the given point (x, y) is inside the polygon (two lists: [list_of_x, list_of_y]).
#     """
#     poly_path = Path(list(zip(polygon[0], polygon[1])))
#     return poly_path.contains_point(point)

def downsample_pointcloud(pointcloud, num_points=1024):
    """
    Downsample the point cloud to the desired number of points using uniform sampling.
    """
    N = pointcloud.shape[0]
    if N >= num_points:
        indices = np.linspace(0, N - 1, num_points, dtype=int)
        return pointcloud[indices]
    else:
        reps = num_points // N + 1
        repeated = np.tile(pointcloud, (reps, 1))
        return repeated[:num_points]

# -------------------------------------------------------------------
# Dataset Class (Pipeline A)
# -------------------------------------------------------------------

class TableClassificationDataset(Dataset):
    """
    Pipeline A: Converts a depth image to a 3D point cloud.
    If table polygon annotation exists, only points that belong to the table are returned;
    otherwise, all valid points (background) are returned.
    Each sample contains:
      - pointcloud: 3D point cloud of (num_points x 3) dimensions
      - label: 1 (table) or 0 (no table)
      - file_path: path to the original depth image
    """
    def __init__(self, root_dir, depth_folder="depth", annotation_path="tabletop_labels.dat",
                 intrinsics_path=None, num_points=1024, transform=None, verbose=False):
        self.root_dir = root_dir
        self.depth_dir = os.path.join(root_dir, depth_folder)
        self.annotation_path = os.path.join(root_dir, annotation_path)
        self.num_points = num_points
        self.transform = transform
        self.verbose = verbose

        # Only PNG files
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.lower().endswith('.png')])
        
        # Filter out blacklisted images
        for sequence, blacklist_images in BLACKLIST.items():
            if sequence in root_dir:
                if self.verbose:
                    print(f"Filtering blacklisted images for sequence: {sequence}")
                before_count = len(self.depth_files)
                self.depth_files = [f for f in self.depth_files if f not in blacklist_images]
                after_count = len(self.depth_files)
                if self.verbose and before_count != after_count:
                    print(f"Filtered out {before_count - after_count} images.")

        # Load annotations (pickle file)
        if os.path.exists(self.annotation_path):
            with open(self.annotation_path, 'rb') as f:
                self.annotations = pickle.load(f)
            if self.verbose:
                print(f"Loaded {len(self.annotations)} annotations from {self.annotation_path}.")

            # If we filtered out blacklisted images, we need to adjust annotations accordingly
            if any(sequence in root_dir for sequence in BLACKLIST):
                # Get original filenames before filtering
                original_files = sorted([f for f in os.listdir(self.depth_dir) if f.lower().endswith('.png')])
                
                # Create mapping from original index to new index
                file_to_index = {file: i for i, file in enumerate(original_files)}
                
                # Keep only annotations for files that weren't blacklisted
                filtered_annotations = []
                for i, file in enumerate(self.depth_files):
                    original_idx = file_to_index.get(file)
                    if original_idx is not None and original_idx < len(self.annotations):
                        filtered_annotations.append(self.annotations[original_idx])
                
                self.annotations = filtered_annotations
                if self.verbose:
                    print(f"Adjusted annotations after filtering blacklisted images: {len(self.annotations)} entries")
        else:
            print(f"Annotation file not found at {self.annotation_path}. Using empty annotations.")
            self.annotations = [[] for _ in range(len(self.depth_files))]

        self.intrinsics = load_intrinsics(intrinsics_path)

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        # Load depth image
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        depth_img = depth_img.astype(np.float32)

        # Calculate 3D point cloud and pixel coordinates from depth
        pointcloud, pixel_coords = depth_to_pointcloud(depth_img, self.intrinsics)
        
        # Get annotation (e.g., [[list_of_x, list_of_y], ...])
        annotation = self.annotations[idx]
        
        if len(annotation) > 0:
            # Table annotation exists: Select only points that belong to the table
            table_mask = np.zeros(pixel_coords.shape[0], dtype=bool)
            for poly in annotation:
                p = Path(list(zip(poly[0], poly[1])))
                mask = p.contains_points(pixel_coords)
                table_mask |= mask
            filtered_pointcloud = pointcloud[table_mask]
            label = 1
        else:
            # No annotation: Use all valid points
            filtered_pointcloud = pointcloud
            label = 0

        # Downsample to desired number of points
        sampled_pointcloud = downsample_pointcloud(filtered_pointcloud, self.num_points)
        sampled_pointcloud = torch.from_numpy(sampled_pointcloud).float()
        label = torch.tensor(label, dtype=torch.long)

        sample = {
            "pointcloud": sampled_pointcloud,
            "label": label,
            "file_path": depth_path
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

# -------------------------------------------------------------------
# Optional: Random data augmentation function
# -------------------------------------------------------------------
def random_augmentation(sample):
    """Apply random augmentation to point cloud data."""
    pointcloud = sample["pointcloud"].numpy()
    angle = np.random.uniform(0, 2 * np.pi)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    pointcloud = pointcloud @ R.T
    scale = np.random.uniform(0.9, 1.1)
    pointcloud *= scale
    jitter = np.random.normal(0, 0.01, pointcloud.shape)
    pointcloud += jitter
    sample["pointcloud"] = torch.from_numpy(pointcloud).float()
    return sample

# -------------------------------------------------------------------
# Visualize Function: Show depth image and point cloud side by side.
# -------------------------------------------------------------------
def visualize_sample(sample):
    """
    Visualize the depth image and point cloud side by side.
    """
    depth_img = cv2.imread(sample["file_path"], cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError("Depth image not found for visualization.")
    pc = sample["pointcloud"].numpy()

    fig = plt.figure(figsize=(12, 6))
    
    # Left side: Depth image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(depth_img, cmap='gray')
    ax1.set_title("Depth Image")
    ax1.axis('off')
    
    # Right side: 3D point cloud scatter plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=pc[:, 2], cmap='viridis')
    ax2.set_title("3D Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# Compute class weights for class imbalance
# -------------------------------------------------------------------   

def compute_class_weights(dataset):
    """
    Compute weights for each class to handle class imbalance.
    Weights are inversely proportional to class frequencies.
    """
    # Count occurrences of each class
    class_counts = {0: 0, 1: 0}
    for i in range(len(dataset)):
        label = dataset[i]["label"].item()
        class_counts[label] += 1
    
    # Calculate weights (inversely proportional to frequency)
    total_samples = sum(class_counts.values())
    weights = {
        label: total_samples / (len(class_counts) * count) 
        for label, count in class_counts.items()
    }
    
    print("Class distribution:")
    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples ({count/total_samples:.2%})")
    
    print("Class weights:")
    for label, weight in weights.items():
        print(f"Class {label} weight: {weight:.4f}")
    
    return torch.FloatTensor([weights[0], weights[1]])

# -------------------------------------------------------------------
# Create Train and Test Dataset using sequences
# -------------------------------------------------------------------
def process_sequences(seq_list, base_path, set_name="Set", transform=None):
    dataset_list = []
    for seq in seq_list:
        seq_path = os.path.join(base_path, seq)
        # If "harvard_tea_2" sequence exists, use "depth" folder; otherwise "depthTSDF"
        # if "harvard_tea_2" in seq:
        #     depth_folder = "depth"
        # else:
        #     depth_folder = "depthTSDF"
        depth_folder = "depth_pred"
        annotation_path = "labels/tabletop_labels.dat"
        intrinsics_path = os.path.join(seq_path, "intrinsics.txt")
        ds = TableClassificationDataset(root_dir=seq_path,
                                          depth_folder=depth_folder,
                                          annotation_path=annotation_path,
                                          intrinsics_path=intrinsics_path,
                                          num_points=1024,
                                          transform=transform,
                                          verbose=True)
        dataset_list.append(ds)
        print(f"Sequence '{seq}' contains {len(ds)} samples.")
    combined = ConcatDataset(dataset_list) if dataset_list else None
    if combined is not None:
        print(f"{set_name} dataset contains: {len(combined)} samples.")
    else:
        print("No samples found for", set_name)
    return combined

# -------------------------------------------------------------------
# Main Program: Create Train and Test Datasets and visualize examples
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Sequences definitions
    sequences_train = [
        "mit_32_d507/d507_2",
        "mit_76_459/76-459b",
        "mit_76_studyroom/76-1studyroom2",
        "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika",  # negative samples
        "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"
    ]
    
    sequences_test = [
        "harvard_c5/hv_c5_1",
        "harvard_c6/hv_c6_1",
        "harvard_c11/hv_c11_2",
        "harvard_tea_2/hv_tea2_2"  # Raw depth images are in "depth" folder here. --negative samples
    ]
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two directories to reach project root
    base_path = os.path.normpath(os.path.join(script_dir, "../../data/CW2-Dataset/"))
    
    # Data augmentation can be applied to train datasets.
    train_dataset = process_sequences(sequences_train, base_path, set_name="Train", transform=random_augmentation)


    print("--------------------------------------------------------------------------------")

    # No data augmentation for test datasets.
    test_dataset = process_sequences(sequences_test, base_path, set_name="Test", transform=None)
    
    # Take an example from the created train dataset and show its information and visualize it.
    if len(train_dataset) > 0:
        sample = train_dataset[2]
        print("Sample keys:", sample.keys())
        print("Point cloud shape:", sample["pointcloud"].shape)
        print("Label (1: table, 0: no table):", sample["label"])
        print("File path:", sample["file_path"])
        visualize_sample(sample)
