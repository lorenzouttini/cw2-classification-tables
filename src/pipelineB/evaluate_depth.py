#!/usr/bin/env python3
import os
import numpy as np
import torch
from PIL import Image
import argparse
from transformers import DPTImageProcessor, DPTForDepthEstimation
import matplotlib.pyplot as plt


# ----------- Check GT Depths -----------

def load_groundtruth_depth(png_path):
    # Load as grayscale / "F" for 32-bit float
    gt = Image.open(png_path).convert("I")  # "I" keeps 32-bit int if it's a 16-bit PNG
    gt = np.array(gt, dtype=np.float32)
    print(f"GT min: {gt.min()}, max: {gt.max()}")

    # Convert SUN dataset GT from millimeters to meters
    gt = gt / 1000.0

    # Mask out invalid pixels (e.g. zeros)
    mask = (gt > 1e-6)
    return gt, mask

def resize_depth_map(depth, target_shape):
    depth_img = Image.fromarray(depth)
    depth_resized = depth_img.resize((target_shape[1], target_shape[0]), Image.BICUBIC)
    return np.array(depth_resized, dtype=np.float32)

# ========== Load DPT Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device).eval()

def process_depth(scene_path, output_dir, max_frames=100):
    """
    For each RGB image in the scene folder, compute the depth map using DPT,
    remove outliers and normalize the result, and save it as a .npy file.
    """
    image_dir = os.path.join(scene_path, "image")
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(image_dir))[:max_frames]
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Run DPT to predict depth
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            depth = depth_model(**inputs).predicted_depth[0].cpu().numpy()
        
        # Remove outliers and normalize the depth map
        d_min, d_max = np.percentile(depth, 1), np.percentile(depth, 99)
        depth = np.clip(depth, d_min, d_max)
        # depth = (depth - d_min) / (d_max - d_min)
        
        # Save the depth map as a .npy file
        out_file = os.path.join(output_dir, f"{os.path.basename(scene_path)}_{idx:04d}_depth.npy")
        np.save(out_file, depth)
        print(f"[{idx+1}/{len(image_files)}] Saved depth map: {out_file}")


# ----------- Compute Depth Metrics ------------
def compute_depth_metrics(pred_depth, gt_depth, valid_mask):
    """
    Compute depth metrics: absolute relative error and RMSE.
    """
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    abs_rel = np.mean(np.abs(pred_depth - gt_depth) / gt_depth)
    rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))

    return abs_rel, rmse


def plot_depths(gt, pred, mask, out_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap='plasma')
    plt.title("GT Depth")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='plasma')
    plt.title("Predicted Depth")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(gt - pred) * mask, cmap='inferno')
    plt.title("Abs Error")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



# ----------- Main Function ------------

if __name__ == "__main__":
    # Compute depth maps
    scene_path = "data/mit_76_459/76-459b"
    output_path = "data/depth_maps"
    max_frames = 150  
    process_depth(scene_path, output_path, max_frames)

    # Load pred depth
    pred_depth_path = os.path.join(output_path, f"{os.path.basename(scene_path)}_0000_depth.npy")
    pred_depth = np.load(pred_depth_path)
    print(f"Predicted depth shape: {pred_depth.shape}")

    # Load GT depth
    gt_depth, mask = load_groundtruth_depth("data/mit_76_459/76-459b/depthTSDF/0000001-000000028949.png")
    print(f"GT depth shape: {gt_depth.shape}")

    # Resize GT depth if different from pred depth
    if pred_depth.shape != gt_depth.shape:
        pred_depth = resize_depth_map(pred_depth, gt_depth.shape)
        print(f"Resized predicted depth shape: {pred_depth.shape}")
    

    #Combine mask with pred validity (pred_depth>0)
    # Mask: valid where depth is > 0
    valid_mask = gt_depth > 0
    valid_mask = valid_mask & (pred_depth > 0)

    # Align depth scaling
    pred_depth = 1.0 / (pred_depth + 1e-8)
    scale = np.median(gt_depth[valid_mask]) / (np.median(pred_depth[valid_mask]) + 1e-8)
    pred_depth_aligned = pred_depth * scale
 

    abs_rel, rmse = compute_depth_metrics(pred_depth_aligned, gt_depth, valid_mask)
    print("Abs Rel:", abs_rel, "RMSE:", rmse)
    plot_depths(gt_depth, pred_depth_aligned, valid_mask, "figures/results/depths/depth_comparison_0001.png")
