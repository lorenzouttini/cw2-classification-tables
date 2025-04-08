#!/usr/bin/env python3
import os
import numpy as np
import pickle

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

def find_scene_folder(subfolder_name, base_dir="data"):
    """
    Recursively search in base_dir for a folder named subfolder_name that has 'intrinsics.txt'.
    Returns the full path if found, otherwise None.
    """
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == subfolder_name and "intrinsics.txt" in files:
            return root
    return None

def load_intrinsics(path):
    """
    Loads a 3x3 camera intrinsics matrix from intrinsics.txt.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    mat = []
    for line in lines:
        mat.append([float(val) for val in line.strip().split()])
    return np.array(mat)

def depth_to_pointcloud(depth, intrinsics):
    """
    Convert a depth map to a point cloud using the intrinsics.
    """
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth
    x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def has_table(polygon_list):
    return len(polygon_list) > 0

def convert_depth_maps_to_pointclouds(depth_dir, pc_output_dir, base_data_dir="data"):
    """
    For each .npy depth map in depth_dir:
      1) Parse subfolder_name from the filename (e.g. "hv_c6_1_0000_depth.npy" -> "hv_c6_1")
      2) Find that subfolder in 'data/' to load intrinsics.txt and the labels.
      3) Convert depth -> pointcloud.
      4) Match the filename prefix with 'image/*.jpg' to get the index of the label (e.g. 0000 from 0000-12345.jpg).
      5) Use that index to retrieve the polygon list from 'tabletop_labels.dat'.
      6) Save a dict with keys "points" and "label" as .npy with _pc.npy suffix.
    """
    os.makedirs(pc_output_dir, exist_ok=True)

    label_cache = {}
    image_file_index_cache = {}

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith("_depth.npy")])
    for depth_file in depth_files:
        depth_path = os.path.join(depth_dir, depth_file)

        parts = depth_file.split("_")
        if len(parts) < 3:
            print(f"Skipping {depth_file}, not enough '_' to parse subfolder name.")
            continue

        subfolder_name = "_".join(parts[:-2])
        scene_folder = find_scene_folder(subfolder_name, base_data_dir)
        if scene_folder is None:
            print(f" Warning: Could not find scene folder for '{subfolder_name}'. Skipping {depth_file}.")
            continue

        intrinsics_path = os.path.join(scene_folder, "intrinsics.txt")
        if not os.path.exists(intrinsics_path):
            print(f"Warning: No intrinsics.txt in {scene_folder}. Skipping {depth_file}.")
            continue

        intrinsics = load_intrinsics(intrinsics_path)
        depth = np.load(depth_path)
        points = depth_to_pointcloud(depth, intrinsics)

        frame_prefix = parts[-2].zfill(7)  # Normalize to match image file keys
        # print(f"\n Processing: {depth_file} | Frame prefix (padded): {frame_prefix}")

        # Load label cache if not already
        if scene_folder not in label_cache:
            label_path = os.path.join(scene_folder, "labels", "tabletop_labels.dat")
            if os.path.exists(label_path):
                with open(label_path, 'rb') as f:
                    label_cache[scene_folder] = pickle.load(f)
                # Build image filename map
                image_dir = os.path.join(scene_folder, "image")
                image_files = sorted(os.listdir(image_dir))
                #print(f"Loaded {len(image_files)} image files from: {image_dir}")
                #print(f" First 5 image filename keys:")
                # for img in image_files[:5]:
                #     print("   ", img)

                image_file_index_cache[scene_folder] = {
                    os.path.splitext(img)[0].split("-")[0]: idx
                    for idx, img in enumerate(image_files)
                }

                #print(f"Built image_file_index_cache with {len(image_file_index_cache[scene_folder])} keys")
                #print(f"Sample keys: {list(image_file_index_cache[scene_folder].keys())[:5]}")
            else:
                label_cache[scene_folder] = None
                image_file_index_cache[scene_folder] = {}
                print(f" Warning: No labels found in {label_path}. All frames in this scene will be labeled as 0.")

        label_polygons = label_cache[scene_folder]
        index_lookup = image_file_index_cache[scene_folder]

        # Check if image is blacklisted
        scene_key = os.path.basename(os.path.dirname(scene_folder))  # e.g., mit_32_d507
        image_name = next((img for img in index_lookup if img.startswith(frame_prefix)), None)
        if image_name and scene_key in BLACKLIST and image_name + ".jpg" in BLACKLIST[scene_key]:
            print(f"⏭️  Skipping blacklisted image: {image_name}.jpg in scene {scene_key}")
            continue

        if label_polygons is not None:
            label_idx = index_lookup.get(frame_prefix)
            if label_idx is not None and label_idx < len(label_polygons):
                polygons = label_polygons[label_idx]
                label = int(has_table(polygons))
                # print(f" Matched label at index {label_idx}, table present: {bool(label)}")
            else:
                #print(f" Could not match label for frame prefix '{frame_prefix}' in {scene_folder}.")
                #print(f"   Available keys: {list(index_lookup.keys())[:10]}")
                label = 0
        else:
            label = 0

        sample = {
            "points": points.astype(np.float32),
            "label": label
        }

        out_path = os.path.join(pc_output_dir, depth_file.replace("_depth.npy", "_pc.npy"))
        np.save(out_path, sample)
        print(f" Saved point cloud: {out_path} (label={label})")


def main():
    depth_dir = "data/depth_maps"
    pc_output_dir = "data/pointclouds"
    base_data_dir = "data"

    convert_depth_maps_to_pointclouds(depth_dir, pc_output_dir, base_data_dir)

if __name__ == "__main__":
    main()

