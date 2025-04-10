#!/usr/bin/env python3
import os
import sys
import re
import numpy as np
import torch
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipelineB.utils.depths import load_groundtruth_depth_custom, resize_depth_map, load_intrinsics
from pipelineB.utils.visualization import plot_depths_with_mask
from pipelineB.utils.metrics import compute_depth_metrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize the image processor and load the depth model.
# FOR UCL DATASET, we use CPU.
processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
depth_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to("cpu").eval()
device = torch.device("cpu")

def process_depth(scene_path, output_dir, max_frames=100):
    """
    For each RGB image in the scene folder, compute the depth map using ZoeDepth,
    remove outliers and normalize it, and save as a .npy file.
    """
    image_dir = os.path.join(scene_path, "rgb")
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(image_dir))[:max_frames]
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Run depth model.
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            depth = depth_model(**inputs).predicted_depth[0].cpu().numpy()
        
        # Remove outliers and normalize.
        d_min, d_max = np.percentile(depth, 3), np.percentile(depth, 97)
        depth = np.clip(depth, d_min, d_max)

        # FOR UCL DATASET: extract frame number from filename.
        match = re.search(r"(\d+)", img_name)
        if match:
            frame_num = int(match.group(1))
        else:
            raise ValueError(f"Could not extract frame number from filename: {img_name}")
        out_file = os.path.join(output_dir, f"{os.path.basename(scene_path)}_{frame_num:03d}_depth.npy")
        
        np.save(out_file, depth)
        print(f"[{idx+1}/{len(image_files)}] Saved depth map: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Process depth maps for custom UCL dataset.")
    # Set default folders_type to "ucl" for your custom dataset.
    parser.add_argument("--folders_type", type=str, default="ucl",
                        choices=["ucl"], help="Type of folder to process.")
    parser.add_argument("--max_frames", type=int, default=250, help="Maximum frames to process per scene.")
    args = parser.parse_args()
    
    # Define paths relative to the project root.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "RealSense"))
    output_path = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "RealSense", "depth_maps"))
    # Define results folder inside src/pipelineB.
    results_root = os.path.abspath(os.path.join(current_dir, "results", "depths", args.folders_type))
    os.makedirs(os.path.join(results_root, "fig"), exist_ok=True)
    
    max_frames = args.max_frames

    # Lists for storing evaluation metrics across images.
    absrel_list = []
    rmse_list = []
    mae_list = []
    log10_list = []

    # Loop over folders starting with the given folders_type.
    ucl_folders = sorted([f for f in os.listdir(base_path) if f.lower().startswith(args.folders_type)])
    print(f"Found {args.folders_type} folders: {ucl_folders}")

    for folder_name in ucl_folders:
        scene_path = os.path.join(base_path, folder_name)
        # For UCL data, do not enforce a single subfolder.
        if args.folders_type != "ucl":
            subfolders = [f for f in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, f))]
            if len(subfolders) == 1:
                scene_path = os.path.join(scene_path, subfolders[0])
            else:
                print(f"  Skipping {folder_name} — expected 1 subfolder, found {len(subfolders)}.")
                continue
        # Else, for UCL, use the folder as is.
        if not os.path.isdir(scene_path):
            continue

        print(f"\n=== Processing Scene: {scene_path} ===")
        # 1) Generate predicted depth maps.
        process_depth(scene_path, output_path, max_frames)

        # 2) Load predictions and ground-truth depth, then evaluate.
        image_dir = os.path.join(scene_path, "rgb")
        depth_dir = os.path.join(scene_path, "depth")
        if not (os.path.isdir(image_dir) and os.path.isdir(depth_dir)):
            print(f"  Skipping {scene_path} (missing rgb/ or depth/).")
            continue

        image_files = sorted(os.listdir(image_dir))[:max_frames]
        depth_files = sorted(os.listdir(depth_dir))[:max_frames]

        for img_file, gt_file in zip(image_files, depth_files):
            img_path = os.path.join(image_dir, img_file)
            gt_path = os.path.join(depth_dir, gt_file)

            # Extract frame number.
            match = re.search(r"(\d+)", img_file)
            if match:
                frame_num = int(match.group(1))
            else:
                print(f" Warning: Could not extract frame number from filename '{img_file}', skipping.")
                continue

            subfolder_name = os.path.basename(scene_path)
            pred_depth_path = os.path.join(output_path, f"{subfolder_name}_{frame_num:03d}_depth.npy")
            if not os.path.exists(pred_depth_path):
                print(f"  Warning: No predicted depth found for {img_file}, skipping.")
                continue

            pred_depth = np.load(pred_depth_path)
            # Load ground truth depth & associated mask.
            gt_depth, mask = load_groundtruth_depth_custom(gt_path)
            if pred_depth.shape != gt_depth.shape:
                pred_depth = resize_depth_map(pred_depth, gt_depth.shape)

            # Create a valid mask.
            valid_mask = (gt_depth > 0.04) & (gt_depth < 0.30) & (pred_depth > 0.01)

            # Align predicted depth to GT scale.
            pred_med = np.median(pred_depth[valid_mask]) if np.any(valid_mask) else 1e-8
            gt_med   = np.median(gt_depth[valid_mask])   if np.any(valid_mask) else 1e-8
            scale = gt_med / (pred_med + 1e-8)
            pred_depth_aligned = pred_depth * scale

            # Compute metrics.
            metrics = compute_depth_metrics(pred_depth_aligned, gt_depth, valid_mask)
            absrel_list.append(metrics["AbsRel"])
            rmse_list.append(metrics["RMSE"])
            if "MAE" in metrics:
                mae_list.append(metrics["MAE"])
            if "Log10" in metrics:
                log10_list.append(metrics["Log10"])
            
            # Save depth comparison plot.
            plot_path = os.path.join(results_root, "fig", f"{folder_name}_{frame_num:03d}_depth_comparison_masked.png")
            plot_depths_with_mask(gt_depth, pred_depth_aligned, valid_mask, plot_path)
            
            # Save png visualization.
            pred_dir = os.path.join(scene_path, "depth_pred")
            os.makedirs(pred_dir, exist_ok=True)
            vis = (pred_depth_aligned - pred_depth_aligned.min()) / (pred_depth_aligned.max() - pred_depth_aligned.min() + 1e-8)
            vis_img = (vis * 255).astype(np.uint8)
            Image.fromarray(vis_img).save(os.path.join(pred_dir, f"{subfolder_name}_{frame_num:03d}_depth.png"))

    # Compute overall averaged metrics.
    avg_absrel = np.mean(absrel_list) if absrel_list else 0
    avg_rmse   = np.mean(rmse_list)   if rmse_list   else 0
    avg_mae    = np.mean(mae_list)    if mae_list    else 0
    avg_log10  = np.mean(log10_list)  if log10_list  else 0

    print("\n=== Averaged Depth Evaluation Metrics (all {} folders) ===".format(args.folders_type))
    print(f"AbsRel: {avg_absrel:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"MAE: {avg_mae:.4f}")
    print(f"Log10: {avg_log10:.4f}")
    print(f"Total images processed: {len(absrel_list)}")
    print(f"Total folders processed: {len(ucl_folders)}")

    # Save averaged metrics to a text file.
    results_file = os.path.join(results_root, "results_depth.txt")
    with open(results_file, "w") as f:
        f.write("Averaged Depth Evaluation Metrics (all {} folders):\n".format(args.folders_type))
        f.write(f"AbsRel: {avg_absrel:.4f}\n")
        f.write(f"RMSE: {avg_rmse:.4f}\n")
        f.write(f"MAE: {avg_mae:.4f}\n")
        f.write(f"Log10: {avg_log10:.4f}\n")
    print(f"\nSaved averaged metrics to {results_file}")

    # Plot and save histogram for AbsRel.
    plt.figure(figsize=(6, 4))
    plt.hist(absrel_list, bins=30, color="cornflowerblue", edgecolor="black")
    plt.title(f"AbsRel Histogram (All {args.folders_type} images)")
    plt.xlabel("AbsRel")
    plt.ylabel("Frequency")
    plt.tight_layout()
    hist_path = os.path.join(results_root, "absrel_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved AbsRel histogram to {hist_path}")

if __name__ == "__main__":
    main()