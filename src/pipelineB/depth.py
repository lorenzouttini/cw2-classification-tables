import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ZoeDepthForDepthEstimation, AutoImageProcessor
import sys
import argparse
import re

# Add parent directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipelineB.utils.visualization import plot_depths_with_mask
from pipelineB.utils.metrics import compute_depth_metrics
from pipelineB.utils.depths import load_groundtruth_depth, resize_depth_map, load_intrinsics
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------- Load New Depth Model (e.g., ZoeDepth) ----------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
depth_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(device).eval()


def process_depth(scene_path, output_dir, max_frames=100):
    """
    For each RGB image in the scene folder, compute the depth map using the depth model,
    remove outliers and normalize the result, and save it as a .npy file.
    """
    image_dir = os.path.join(scene_path, "image")
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for common image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])[:max_frames]
    
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            depth = depth_model(**inputs).predicted_depth[0].cpu().numpy()

        # Remove outliers and normalize the depth map
        d_min, d_max = np.percentile(depth, 3), np.percentile(depth, 97)
        depth = np.clip(depth, d_min, d_max)
        
        # Save the depth map as a .npy file
        frame_index = img_name.split("-")[0]
        out_file = os.path.join(output_dir, f"{os.path.basename(scene_path)}_{int(frame_index):04d}_depth.npy")
        np.save(out_file, depth)
        print(f"[{idx+1}/{len(image_files)}] Saved depth map: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Process depth maps with dynamic folder settings.")
    parser.add_argument("--folders_type", type=str, default="mit",
                        choices=["mit", "harvard"], help="Type of folder to process.")
    parser.add_argument("--max_frames", type=int, default=250, help="Maximum frames to process per scene.")
    args = parser.parse_args()

    # Define data directory relative to project root.
    # Assumes data folder exists at the same level as src.
    base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "CW2-Dataset"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "CW2-Dataset", "depth_maps"))
    
    # Define results folder inside pipelineB directory.
    results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    absrel_list = []
    rmse_list = []
    mae_list = []
    log10_list = []
    
    folders = sorted([f for f in os.listdir(base_path) if f.startswith(args.folders_type)])
    print(f"Found {args.folders_type} folders: {folders}")
    
    for folder_name in folders:
        scene_path = os.path.join(base_path, folder_name)
        subfolders = [f for f in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, f))]
        if len(subfolders) == 1:
            scene_path = os.path.join(scene_path, subfolders[0])
        else:
            print(f"  Skipping {folder_name} — expected 1 subfolder, found {len(subfolders)}.")
            continue

        if not os.path.isdir(scene_path):
            continue

        print(f"\n=== Processing Scene: {scene_path} ===")
        process_depth(scene_path, output_path, args.max_frames)

        image_dir = os.path.join(scene_path, "image")
        if (os.path.basename(os.path.dirname(scene_path)) == "harvard_tea_2" and
            os.path.basename(scene_path) == "hv_tea2_2"):
            depth_dir = os.path.join(scene_path, "depth")
        else: 
            depth_dir = os.path.join(scene_path, "depthTSDF")

        if not (os.path.isdir(image_dir) and os.path.isdir(depth_dir)):
            print(f"  Skipping {scene_path} (missing image/ or depth directory).")
            continue

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])[:args.max_frames]
        depth_files = sorted(os.listdir(depth_dir))[:args.max_frames]

        for img_file, gt_file in zip(image_files, depth_files):
            img_path = os.path.join(image_dir, img_file)
            gt_path = os.path.join(depth_dir, gt_file)
            frame_index = img_file.split("-")[0]
            if not frame_index.isdigit():
                print(f"  Warning: Filename {img_file} does not start with a numeric frame index, skipping.")
                continue
            frame_num = int(frame_index)
            subfolder_name = os.path.basename(scene_path)
            pred_depth_path = os.path.join(output_path, f"{subfolder_name}_{frame_num:04d}_depth.npy")
            if not os.path.exists(pred_depth_path):
                print(f"  Warning: No predicted depth found for {img_file}, skipping.")
                continue
            pred_depth = np.load(pred_depth_path)
            gt_depth, mask = load_groundtruth_depth(gt_path)
            if pred_depth.shape != gt_depth.shape:
                pred_depth = resize_depth_map(pred_depth, gt_depth.shape)
            valid_mask = (gt_depth > 0.6) & (pred_depth > 0.01)
            pred_med = np.median(pred_depth[valid_mask]) if np.any(valid_mask) else 1e-8
            gt_med   = np.median(gt_depth[valid_mask])   if np.any(valid_mask) else 1e-8
            scale = gt_med / (pred_med + 1e-8)
            pred_depth_aligned = pred_depth * scale

            metrics = compute_depth_metrics(pred_depth_aligned, gt_depth, valid_mask)
            absrel_list.append(metrics["AbsRel"])
            rmse_list.append(metrics["RMSE"])
            if "MAE" in metrics:
                mae_list.append(metrics["MAE"])
            if "Log10" in metrics:
                log10_list.append(metrics["Log10"])

            if args.folders_type == "mit":
                l = "train"
            elif args.folders_type == "harvard":
                l = "test"
            else:
                l = "ucl"

            results_dir = os.path.join(results_root, "depths", l)
            os.makedirs(os.path.join(results_dir, "fig"), exist_ok=True)
            plot_depths_with_mask(
                gt_depth, pred_depth_aligned, valid_mask,
                os.path.join(results_dir, "fig", f"{subfolder_name}_{frame_num:04d}_depth_comparison_masked.png")
            )
            os.makedirs(os.path.join(scene_path, "depth_pred"), exist_ok=True)
            depth_uint16 = (pred_depth_aligned * 10000).astype(np.uint16)
            Image.fromarray(depth_uint16).save(
                os.path.join(scene_path, "depth_pred", f"{subfolder_name}_{frame_num:04d}_depth.png")
            )

    avg_absrel = np.mean(absrel_list) if absrel_list else 0
    avg_rmse   = np.mean(rmse_list)   if rmse_list   else 0
    avg_mae    = np.mean(mae_list)    if mae_list    else 0
    avg_log10  = np.mean(log10_list)  if log10_list  else 0

    print(f"\n=== Averaged Depth Evaluation Metrics (all {args.folders_type} folders) ===")
    print(f"AbsRel: {avg_absrel:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"MAE: {avg_mae:.4f}")
    print(f"Log10: {avg_log10:.4f}")
    print(f"Total images processed: {len(absrel_list)}")
    print(f"Total folders processed: {len(folders)}")

    results_file = os.path.join(results_dir, "results_depth.txt")
    with open(results_file, "w") as f:
        f.write(f"Averaged Depth Evaluation Metrics (all {args.folders_type} folders):\n")
        f.write(f"AbsRel: {avg_absrel:.4f}\n")
        f.write(f"RMSE: {avg_rmse:.4f}\n")
        f.write(f"MAE: {avg_mae:.4f}\n")
        f.write(f"Log10: {avg_log10:.4f}\n")
    print(f"\nSaved averaged metrics to {results_file}")

    plt.figure(figsize=(6, 4))
    plt.hist(absrel_list, bins=30, color="cornflowerblue", edgecolor="black")
    plt.title(f"AbsRel Histogram (All {args.folders_type} images)")
    plt.xlabel("AbsRel")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "absrel_histogram.png"))
    plt.close()
    print("Saved AbsRel histogram to absrel_histogram.png")


if __name__ == "__main__":
    main()