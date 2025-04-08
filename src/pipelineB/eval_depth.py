#!/usr/bin/env python3


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import pickle
from transformers import ZoeDepthImageProcessor, ZoeDepthForDepthEstimation, AutoImageProcessor

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ----------- Check GT Depths ----------- #
def load_groundtruth_depth(png_path):
    # Load as grayscale / "I" to keep the integer values
    gt = Image.open(png_path).convert("I")
    gt = np.array(gt, dtype=np.float32)
    # print(f"GT min: {gt.min()}, max: {gt.max()}")
    # Convert SUN dataset GT from decimillimeters to meters
    gt = gt / 1000.0
    # Mask out invalid pixels (e.g. zeros)
    mask = (gt > 1e-6)
    return gt, mask

def resize_depth_map(depth, target_shape):
    depth_img = Image.fromarray(depth)
    depth_resized = depth_img.resize((target_shape[1], target_shape[0]), Image.BICUBIC)
    return np.array(depth_resized, dtype=np.float32)

# ----------- Load New Depth Model (e.g., ZoeDepth) ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the image processor and model
processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
depth_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(device).eval()

# FOR UCL DATASET
# depth_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to("cpu").eval()
# device = torch.device("cpu")

# ----------- Utils ----------- #
def load_intrinsics(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    intrinsics = []
    for line in lines:
        intrinsics.append([float(val) for val in line.strip().split()])
    return np.array(intrinsics)

def depth_to_pointcloud(depth, intrinsics):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth
    x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def has_table(polygon_list):
    return len(polygon_list) > 0

# ----------- BLACKLIST ----------- #
BLACKLIST = {
    "mit_76_studyroom": ["0002111-000070763319.jpg"],
    "mit_32_d507": ["0004646-000155745519.jpg"],
    "harvard_c11": ["0000006-000000187873.jpg"],
    "mit_lab_hj": [
        "0000281-000011395440.jpg",
        "0000676-000027382572.jpg",
        "0001106-000044777376.jpg",
        "0001326-000053659116.jpg"
    ]
}

# ----------- Main: Process Scene and Generate Depth Maps ----------- #

def process_depth(scene_path, output_dir, max_frames=100):
    """
    For each RGB image in the scene folder, compute the depth map using the new depth model (e.g., ZoeDepth),
    remove outliers and normalize the result, and save it as a .npy file.
    """
    image_dir = os.path.join(scene_path, "image")
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(image_dir))[:max_frames]
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Run new depth model (ZoeDepth)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                depth = depth_model(**inputs).predicted_depth[0].cpu().numpy()

        
        # Remove outliers and normalize the depth map
        d_min, d_max = np.percentile(depth, 3), np.percentile(depth, 97)
        depth = np.clip(depth, d_min, d_max)
        # If you want to work with metric values, you might NOT normalize to [0,1] here.
        # Uncomment the following line to normalize:
        # depth = (depth - d_min) / (d_max - d_min)
        
        # Save the depth map as a .npy file
        frame_index = img_name.split("-")[0]
        out_file = os.path.join(output_dir, f"{os.path.basename(scene_path)}_{int(frame_index):04d}_depth.npy")

        np.save(out_file, depth)
        print(f"[{idx+1}/{len(image_files)}] Saved depth map: {out_file}")



# ----------- Compute Depth Metrics ------------ #
def compute_depth_metrics(pred_depth, gt_depth, valid_mask):
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    # Base errors
    abs_rel = np.mean(np.abs(pred_depth - gt_depth) / gt_depth)
    rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
    mae = np.mean(np.abs(pred_depth - gt_depth))
    rmse_log = np.sqrt(np.mean((np.log(pred_depth + 1e-8) - np.log(gt_depth + 1e-8))**2))
    log10 = np.mean(np.abs(np.log10(pred_depth / gt_depth)))

    # Silog
    log_diff = np.log(pred_depth + 1e-8) - np.log(gt_depth + 1e-8)
    silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))

    # Threshold accuracies
    ratio = np.maximum(pred_depth / gt_depth, gt_depth / pred_depth)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    return {
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "MAE": mae,
        "RMSE_log": rmse_log,
        "Log10": log10,
        "Silog": silog,
        "δ1 (<1.25)": delta1,
        "δ2 (<1.25^2)": delta2,
        "δ3 (<1.25^3)": delta3
    }


# ----------- Plot Depths ----------- #
def plot_depths_with_mask(gt, pred, mask, out_path):

    # Apply mask: show gray where mask is False
    masked_pred = np.copy(pred)
    masked_pred[~mask] = 0.0  # This will render as dark violet in 'plasma'

    # Use standard plasma colormap
    cmap = plt.get_cmap('plasma')

    plt.figure(figsize=(10, 4))

    # GT Depth
    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap='plasma')
    plt.title("GT Depth")
    plt.colorbar()

    # Predicted Depth (masked)
    plt.subplot(1, 3, 2)
    plt.imshow(masked_pred, cmap=cmap)
    plt.title("Predicted Depth (masked)")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(gt - pred) * mask, cmap='inferno')
    plt.title("Abs Error")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_depth_histogram(depth, mask):
    values = depth[mask]
    plt.figure(figsize=(8, 4))
    plt.hist(values.flatten(), bins=100, color='dodgerblue')
    plt.title("GT Depth Histogram")
    plt.xlabel("Depth value (after scaling)")
    plt.ylabel("Pixel count")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("figures/results/depths/depth_histogram.png")
    plt.savefig("figures/results/depths/test/depth_histogram.png")
    

'''
# ----------- Main Function ------------ #
if __name__ == "__main__":
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
    
    # Resize predicted depth if necessary to match GT resolution
    if pred_depth.shape != gt_depth.shape:
        pred_depth = resize_depth_map(pred_depth, gt_depth.shape)
        print(f"Resized predicted depth shape: {pred_depth.shape}")
    
    # Create valid mask
    valid_mask = (gt_depth > 5) 
    valid_mask = valid_mask & (pred_depth > 0.01)
    
    # For now, we use Option 1:
    scale = np.median(gt_depth[valid_mask]) / (np.median(pred_depth[valid_mask]) + 1e-8)
    pred_depth_aligned = pred_depth * scale

    metrics = compute_depth_metrics(pred_depth_aligned, gt_depth, valid_mask)

    # Print all metrics nicely
    print("\n Depth Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    plot_depths(gt_depth, pred_depth_aligned, valid_mask, "figures/results/depths/depth_comparison_0003.png")
    plot_depths_with_mask(gt_depth, pred_depth_aligned, valid_mask, "figures/results/depths/depth_comparison_masked_0003.png")
    plot_depth_histogram(gt_depth, mask)
'''

def main():
    base_path = "data"
    output_path = "data/depth_maps"
    max_frames = 250

    # Create a place to store all metrics from all images
    absrel_list = []
    rmse_list = []
    mae_list = []
    log10_list = []

    # 1) Loop over all "mit_" folders
    # mit_folders = sorted([f for f in os.listdir(base_path) if f.startswith("mit")])
    harvard_folders = sorted([f for f in os.listdir(base_path) if f.startswith("harvard")])
    print(f"Found Harvard folders: {harvard_folders}")

    for folder_name in harvard_folders:
        scene_path = os.path.join(base_path, folder_name)
        # Automatically enter the single subfolder
        subfolders = [f for f in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, f))]
        if len(subfolders) == 1:
            scene_path = os.path.join(scene_path, subfolders[0])
        else:
            print(f"  Skipping {folder_name} — expected 1 subfolder, found {len(subfolders)}.")

        if not os.path.isdir(scene_path):
            continue

        print(f"\n=== Processing Scene: {scene_path} ===")

        # 2) Generate predicted depth maps for this scene
        #    (This calls your model and saves .npy files in output_path)
        process_depth(scene_path, output_path, max_frames)

        # 3) Now loop over the images in the scene to load predictions & GT
        image_dir = os.path.join(scene_path, "image")

        if scene_path == "data/harvard_tea_2/hv_tea2_2":
            depth_dir = os.path.join(scene_path, "depth")
        else: 
            depth_dir = os.path.join(scene_path, "depthTSDF")

        if not (os.path.isdir(image_dir) and os.path.isdir(depth_dir)):
            print(f"  Skipping {scene_path} (missing image/ or depthTSDF/).")
            continue

        image_files = sorted(os.listdir(image_dir))[:max_frames]
        depth_files = sorted(os.listdir(depth_dir))[:max_frames]

        # Safety check if they differ in length
        n_images = min(len(image_files), len(depth_files))

        for img_file, gt_file in zip(image_files, depth_files):
            img_path = os.path.join(image_dir, img_file)
            gt_path = os.path.join(depth_dir, gt_file)

            # # Extract the frame index (e.g., "0000123" from "0000123-000000567890.jpg")
            frame_index = img_file.split("-")[0]  # e.g., '0000123'
            frame_num = int(frame_index)          # Convert to int

            # Build predicted depth path using the frame number
            # pred_depth_path = os.path.join(output_path, f"{folder_name}_{frame_num:04d}_depth.npy")
            subfolder_name = os.path.basename(scene_path)  # e.g., "d507_2"
            pred_depth_path = os.path.join(output_path, f"{subfolder_name}_{frame_num:04d}_depth.npy")


            if not os.path.exists(pred_depth_path):
                print(f"  Warning: No predicted depth found for {img_file}, skipping.")
                continue

            pred_depth = np.load(pred_depth_path)

            # 5) Load GT depth
            gt_depth, mask = load_groundtruth_depth(gt_path)

            # If shapes differ, resize predicted
            if pred_depth.shape != gt_depth.shape:
                pred_depth = resize_depth_map(pred_depth, gt_depth.shape)

            # 6) Create valid mask
            valid_mask = (gt_depth > 4) & (pred_depth > 0.01)

            # 7) Align predicted depth to GT scale
            pred_med = np.median(pred_depth[valid_mask]) if np.any(valid_mask) else 1e-8
            gt_med   = np.median(gt_depth[valid_mask])   if np.any(valid_mask) else 1e-8
            scale = gt_med / (pred_med + 1e-8)
            pred_depth_aligned = pred_depth * scale

            # 8) Compute metrics & accumulate
            metrics = compute_depth_metrics(pred_depth_aligned, gt_depth, valid_mask)
            absrel_list.append(metrics["AbsRel"])
            rmse_list.append(metrics["RMSE"])
            if "MAE" in metrics:
                mae_list.append(metrics["MAE"])
            if "Log10" in metrics:
                log10_list.append(metrics["Log10"])
            
            # 8a) Save depth comparison plot
            os.makedirs(f"figures/results/depths/ucl/fig", exist_ok=True)
            # frame_index = img_file.split("-")[0]
            plot_depths_with_mask(
                gt_depth, pred_depth_aligned, valid_mask,
                os.path.join("figures/results/depths/ucl/fig", f"{folder_name}_{int(frame_index):04d}_depth_comparison_masked.png")
            )

            # 8b) Save png images
            os.makedirs(f"{scene_path}/depth_pred", exist_ok=True)
            # Convert to uint16 and to mm for saving as PNG
            Image.fromarray((pred_depth_aligned * 1000).astype(np.uint16)).save(
                os.path.join(f"{scene_path}/depth_pred", f"{subfolder_name}_{int(frame_index):04d}_depth.png")
                )


    # 9) Averages across all images in all mit_ folders
    avg_absrel = np.mean(absrel_list) if absrel_list else 0
    avg_rmse   = np.mean(rmse_list)   if rmse_list   else 0
    avg_mae    = np.mean(mae_list)   if mae_list    else 0
    avg_log10  = np.mean(log10_list) if log10_list  else 0

    print("\n=== Averaged Depth Evaluation Metrics (all UCL folders) ===")
    print(f"AbsRel: {avg_absrel:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"MAE: {avg_mae:.4f}")
    print(f"Log10: {avg_log10:.4f}")
    print(f"Total images processed: {len(absrel_list)}")
    print(f"Total folders processed: {len(harvard_folders)}")

    # 10) Save results to a text file
    # results_file = "figures/results/depths/train/results_depth.txt"
    results_file = "figures/results/depths/test/results_depth.txt"
    with open(results_file, "w") as f:
        f.write("Averaged Depth Evaluation Metrics (all Harvard folders):\n")
        f.write(f"AbsRel: {avg_absrel:.4f}\n")
        f.write(f"RMSE: {avg_rmse:.4f}\n")
        f.write(f"MAE: {avg_mae:.4f}\n")
        f.write(f"Log10: {avg_log10:.4f}\n")

    print(f"\nSaved averaged metrics to {results_file}")

    # 11) Plot histogram of AbsRel
    plt.figure(figsize=(6, 4))
    plt.hist(absrel_list, bins=30, color="cornflowerblue", edgecolor="black")
    plt.title("AbsRel Histogram (All Harvard images)")
    plt.xlabel("AbsRel")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # plt.savefig("figures/results/depths/train/absrel_histogram.png")
    plt.savefig("figures/results/depths/test/absrel_histogram.png")
    plt.close()
    print("Saved AbsRel histogram to absrel_histogram.png")

if __name__ == "__main__":
    main()

