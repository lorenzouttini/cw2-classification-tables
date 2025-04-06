import os
import csv
import numpy as np
import torch
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation, DPTImageProcessor
import pickle
import argparse

# ========== Load DPT model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device).eval()

# ========== Utils ==========
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

# ========== BLACKLIST ========== #
# To fix with LabelMe annotations
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



# ========== Main ==========

def process_scene(scene_path, output_dir, max_frames=100):
    image_dir = os.path.join(scene_path, "image")
    label_file = os.path.join(scene_path, "labels", "tabletop_labels.dat")
    intrinsics_path = os.path.join(scene_path, "intrinsics.txt")

    intrinsics = load_intrinsics(intrinsics_path)
    os.makedirs(output_dir, exist_ok=True)

    # Gets sorted list of images
    image_files = sorted(os.listdir(image_dir))[:max_frames]
    
    # Check if labels are available
    has_labels = os.path.exists(label_file)
    if has_labels:
        with open(label_file, 'rb') as f:
            label_polygons = pickle.load(f)
    else:
        label_polygons = [[] for _ in image_files]  # Empty list => no tables

    scene_name = os.path.basename(os.path.dirname(scene_path))
    # Loops over the images names and their correposnding polygons annotations
    for idx, (img_name, polygons) in enumerate(zip(image_files, label_polygons)):

        # Skip blacklisted frames
        if scene_name in BLACKLIST and img_name in BLACKLIST[scene_name]:
            print(f"⏭️  Skipping blacklisted frame: {img_name}")
            continue
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Run DPT monocular depth - predict depth
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            depth = depth_model(**inputs).predicted_depth[0].cpu().numpy()

        # Remove outliers (2%) and normalize it
        d_min, d_max = np.percentile(depth, 1), np.percentile(depth, 99)
        depth = np.clip(depth, d_min, d_max)
        depth = (depth - d_min) / (d_max - d_min)

        # Each pixel (x,y,z) in the depth map becomes a 3D point in the camera space
        points = depth_to_pointcloud(depth, intrinsics)   # Output shape: [H*W, 3]

        # Assign label: 1 = table (has polygon), 0 = no table
        label = int(has_table(polygons))

        # Save the point cloud and label as a .npy (numpy) file
        sample = {
            "points": points.astype(np.float32),
            "label": label
        }
        out_file = os.path.join(output_dir, f"{os.path.basename(scene_path)}_{idx:04d}.npy")
        np.save(out_file, sample)
        print(f"[{idx+1}/{len(image_files)}] Saved: {out_file} (label={label})")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--scene", type=str, required=True, help="Path to scene folder (e.g., data/mit_76_459/76-459b)")
    # parser.add_argument("--out", type=str, default="data/pointclouds", help="Where to save point clouds")
    # parser.add_argument("--max_frames", type=int, default=100, help="Limit number of images processed (for dev)")
    # args = parser.parse_args()
    # scene_path = "data/mit_76_459/76-459b"
    scene_path = "data/mit_76_studyroom/76-1studyroom2"
    # scene_path = "data/mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika"
    # scene_path = "data/mit_32_d507/d507_2"
    # scene_path = "data/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"
    output_path = "data/pointclouds"
    max_frames = 150

    # process_scene(args.scene, args.out, args.max_frames)
    process_scene(scene_path, output_path, max_frames)