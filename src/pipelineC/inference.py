#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelineC.models.dgcnn import DGCNN_Seg
from pipelineC.dataset import depth_to_pointcloud
from pipelineC.utils.visualization import visualize_with_open3d

def load_realsense_intrinsics(intrinsics_path):
    """
    Load camera intrinsics from a RealSense intrinsics file.
    """
    with open(intrinsics_path, 'r') as f:
        lines = f.readlines()
    
    width_height_line = [l for l in lines if "Width" in l and "Height" in l][0]
    width = int(width_height_line.split(',')[0].split(':')[1].strip())
    height = int(width_height_line.split(',')[1].split(':')[1].strip())
    
    ppx_ppy_line = [l for l in lines if "PPX" in l and "PPY" in l][0]
    ppx = float(ppx_ppy_line.split(',')[0].split(':')[1].strip())
    ppy = float(ppx_ppy_line.split(',')[1].split(':')[1].strip())
    
    fx_fy_line = [l for l in lines if "FX" in l and "FY" in l][0]
    fx = float(fx_fy_line.split(',')[0].split(':')[1].strip())
    fy = float(fx_fy_line.split(',')[1].split(':')[1].strip())
    
    return {
        'width': width,
        'height': height,
        'fx': fx,
        'fy': fy,
        'cx': ppx,
        'cy': ppy
    }

class RealSenseDataset(Dataset):
    """Dataset for custom RealSense data without annotations."""
    def __init__(self, root_dir, depth_folder="depth", rgb_folder="rgb", intrinsics_path=None, num_points=1024):
        self.root_dir = root_dir
        self.depth_dir = os.path.join(root_dir, depth_folder)
        self.rgb_dir = os.path.join(root_dir, rgb_folder)
        self.num_points = num_points
        
        # Get all depth files
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.lower().endswith('.png')])
        print(f"Found {len(self.depth_files)} depth images in {self.depth_dir}")
        
        # Try to find matching RGB files
        self.has_rgb = os.path.exists(self.rgb_dir)
        if self.has_rgb:
            self.rgb_files = []
            for depth_file in self.depth_files:
                # Match by filename (you may need to adjust this logic for your specific naming)
                base_name = os.path.splitext(depth_file)[0]
                rgb_matches = [f for f in os.listdir(self.rgb_dir) 
                               if f.startswith(base_name) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if rgb_matches:
                    self.rgb_files.append(rgb_matches[0])
                else:
                    # If no match, just use None
                    self.rgb_files.append(None)
            print(f"Found {sum(1 for f in self.rgb_files if f is not None)} matching RGB images")
        else:
            self.rgb_files = [None] * len(self.depth_files)
            print("No RGB directory found or it's empty")
        
        # Load intrinsics
        if intrinsics_path and os.path.exists(intrinsics_path):
            self.intrinsics = load_realsense_intrinsics(intrinsics_path)
            print(f"Loaded camera intrinsics from {intrinsics_path}")
        else:
            print("No camera intrinsics provided, using defaults")
            # Default intrinsics for D455 (you might want to adjust these)
            self.intrinsics = {
                'width': 640, 'height': 480,
                'fx': 390.73, 'fy': 390.73,
                'cx': 320.08, 'cy': 244.11
            }
    
    def __len__(self):
        return len(self.depth_files)
    
    def __getitem__(self, idx):
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if depth_img is None:
            raise FileNotFoundError(f"Could not read depth image: {depth_path}")
        
        # Load RGB image if available
        rgb_img = None
        if self.rgb_files[idx]:
            rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is not None:
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Pre-processing: filter invalid depth values (adjust ranges as needed)
        depth_mask = (depth_img > 0) & (depth_img < 5000)  # 0m to 5m range
        filtered_depth = depth_img.copy()
        filtered_depth[~depth_mask] = 0
        
        # Noise reduction
        kernel = np.ones((3, 3), np.uint8)
        filtered_depth = cv2.morphologyEx(filtered_depth, cv2.MORPH_OPEN, kernel)
        filtered_depth = cv2.medianBlur(filtered_depth, 5)  # Remove salt-and-pepper noise
        
        # Convert depth to meters (adjust scaling factor based on your depth units)
        filtered_depth = filtered_depth.astype(np.float32) / 1000.0
        
        # Ensure depth image is 2D single-channel
        if len(filtered_depth.shape) > 2:
            filtered_depth = filtered_depth[:, :, 0]
        
        # Convert intrinsics to matrix format
        intrinsics_matrix = np.array([
            [self.intrinsics['fx'], 0, self.intrinsics['cx']],
            [0, self.intrinsics['fy'], self.intrinsics['cy']],
            [0, 0, 1]
        ])
        
        # Convert depth to point cloud
        pointcloud, pixel_coords = depth_to_pointcloud(filtered_depth, intrinsics_matrix)
        
        # Normalization: center and scale point cloud
        if len(pointcloud) > 0:
            centroid = np.mean(pointcloud, axis=0)
            pointcloud = pointcloud - centroid
            max_dist = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
            if max_dist > 0:
                pointcloud = pointcloud / max_dist
        
        # Downsample to fixed number of points
        if len(pointcloud) > self.num_points:
            indices = np.random.choice(len(pointcloud), self.num_points, replace=False)
            pointcloud = pointcloud[indices]
            pixel_coords = pixel_coords[indices]
        elif len(pointcloud) < self.num_points:
            if len(pointcloud) == 0:
                # Handle completely empty point clouds
                pointcloud = np.zeros((self.num_points, 3))
                pixel_coords = np.zeros((self.num_points, 2))
            else:
                # Repeat some points to reach desired count
                indices = np.random.choice(len(pointcloud), self.num_points, replace=True)
                pointcloud = pointcloud[indices]
                pixel_coords = pixel_coords[indices]
        
        pointcloud = torch.from_numpy(pointcloud).float()
        
        return {
            "pointcloud": pointcloud,
            "pixel_coords": pixel_coords,
            "depth_img": filtered_depth,
            "rgb_img": rgb_img,
            "filename": self.depth_files[idx]
        }

def visualize_prediction(depth_img, rgb_img, pointcloud, pred, pixel_coords, filename, save_path=None):
    """
    Visualize the prediction results with the original image and segmentation results.
    """
    # Create figure with either 2 or 3 subplots depending on whether RGB is available
    if rgb_img is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Normalize and colorize depth image
    depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # RGB image (if available)
    subplot_idx = 0
    if rgb_img is not None:
        axs[subplot_idx].imshow(rgb_img)
        axs[subplot_idx].set_title("RGB Image", fontsize=16)
        axs[subplot_idx].axis('off')
        subplot_idx += 1
    
    # Depth image
    axs[subplot_idx].imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
    axs[subplot_idx].set_title("Depth Image", fontsize=16)
    axs[subplot_idx].axis('off')
    subplot_idx += 1
    
    # Prediction visualization on depth image
    pred_np = pred.cpu().numpy()
    
    # Create a mask for table points
    height, width = depth_img.shape
    table_mask = np.zeros((height, width), dtype=np.uint8)
    
    # For each point, if predicted as table (1), mark the corresponding pixel
    for i, (x, y) in enumerate(pixel_coords.astype(np.int32)):
        if i < len(pred_np) and pred_np[i] == 1:  # If the point is predicted as table
            if 0 <= y < height and 0 <= x < width:
                table_mask[y, x] = 255
    
    # Dilate the mask for better visibility
    table_mask = cv2.dilate(table_mask, np.ones((5,5), np.uint8), iterations=2)
    
    # Create a colorized overlay
    overlay = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB).copy()
    overlay[table_mask > 0] = [255, 0, 0]  # Mark table points in red
    
    axs[subplot_idx].imshow(overlay)
    axs[subplot_idx].set_title("Table Segmentation", fontsize=16)
    axs[subplot_idx].axis('off')
    
    # Add stats to the plot
    table_ratio = np.sum(pred_np == 1) / len(pred_np) * 100
    plt.figtext(0.5, 0.01, f"Table points: {np.sum(pred_np == 1)}/{len(pred_np)} ({table_ratio:.1f}%)", 
                ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.suptitle(f"Table Segmentation: {filename}", fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig

def parse_args():
    parser = argparse.ArgumentParser(description='Run table segmentation inference on custom RealSense data')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained DGCNN model')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to RealSense data folder')
    parser.add_argument('--depth_folder', type=str, default='depth', 
                       help='Name of the folder containing depth images')
    parser.add_argument('--rgb_folder', type=str, default='rgb', 
                       help='Name of the folder containing RGB images')
    parser.add_argument('--intrinsics', type=str, default=None, 
                       help='Path to camera intrinsics file')
    parser.add_argument('--save_dir', type=str, default='./src/pipelineC/results/custom_inference', 
                       help='Directory to save results')
    parser.add_argument('--num_points', type=int, default=1024, 
                       help='Number of points for point cloud sampling')
    parser.add_argument('--visualize_3d', action='store_true', 
                       help='Show interactive 3D visualization with Open3D')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Look for intrinsics file if not specified
    if args.intrinsics is None:
        default_intrinsics = os.path.join(args.data_dir, "camera_intrinsics.txt")
        if os.path.exists(default_intrinsics):
            args.intrinsics = default_intrinsics
            print(f"Found and using intrinsics file: {args.intrinsics}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = DGCNN_Seg(k=20, num_classes=2, emb_dims=1024)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create dataset
    dataset = RealSenseDataset(
        root_dir=args.data_dir,
        depth_folder=args.depth_folder,
        rgb_folder=args.rgb_folder,
        intrinsics_path=args.intrinsics,
        num_points=args.num_points
    )
    
    # Process each sample
    table_points_ratio = []
    print(f"\nProcessing {len(dataset)} images...")
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        pointcloud = sample["pointcloud"]
        filename = sample["filename"]
        depth_img = sample["depth_img"]
        rgb_img = sample["rgb_img"]
        pixel_coords = sample["pixel_coords"]
        
        print(f"[{idx+1}/{len(dataset)}] Processing: {filename}")
        
        # Skip if point cloud is empty
        if torch.all(pointcloud == 0):
            print(f"  Warning: Empty point cloud for {filename}, skipping")
            continue
        
        # Run inference
        with torch.no_grad():
            input_pc = pointcloud.unsqueeze(0).to(device)
            output = model(input_pc)
            pred = output.max(dim=2)[1].squeeze(0)
        
        # Calculate table percentage
        table_percent = (pred == 1).float().mean().item() * 100
        table_points_ratio.append(table_percent)
        print(f"  Table points: {(pred == 1).sum().item()}/{len(pred)} ({table_percent:.1f}%)")
        
        # Save visualization
        save_path = os.path.join(args.save_dir, f"{os.path.splitext(filename)[0]}_segmentation.png")
        visualize_prediction(
            depth_img, rgb_img, pointcloud, pred, pixel_coords, 
            filename, save_path=save_path
        )
        print(f"  Saved visualization to {save_path}")
        
        # Optional: 3D visualization with Open3D
        if args.visualize_3d:
            visualize_with_open3d(pointcloud, pred, f"{filename} - Table Segmentation")
    
    # Save summary statistics
    print("\nSummary statistics:")
    print(f"  Average table points: {np.mean(table_points_ratio):.2f}%")
    print(f"  Min table points: {np.min(table_points_ratio):.2f}%")
    print(f"  Max table points: {np.max(table_points_ratio):.2f}%")
    
    # Save statistics to a file
    stats_path = os.path.join(args.save_dir, "segmentation_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Total images processed: {len(table_points_ratio)}\n")
        f.write(f"Average table points: {np.mean(table_points_ratio):.2f}%\n")
        f.write(f"Min table points: {np.min(table_points_ratio):.2f}%\n")
        f.write(f"Max table points: {np.max(table_points_ratio):.2f}%\n")
        
        # Add per-image statistics
        f.write("\nPer-image statistics:\n")
        for idx, filename in enumerate(dataset.depth_files[:len(table_points_ratio)]):
            f.write(f"{filename}: {table_points_ratio[idx]:.2f}%\n")
    
    print(f"\nInference complete! Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()