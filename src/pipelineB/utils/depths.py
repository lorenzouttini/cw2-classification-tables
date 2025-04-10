
import os
import numpy as np
import torch
from PIL import Image

def load_groundtruth_depth(png_path):
    # Load as grayscale / "I" to keep the integer values
    gt = Image.open(png_path).convert("I")
    gt = np.array(gt, dtype=np.float32)
    # Convert SUN dataset GT from decimillimeters to meters
    gt = gt / 10000.0
    # Mask out invalid pixels (e.g. zeros)
    mask = (gt > 1e-6)
    return gt, mask

# ----------- Check GT Depths ----------- #
def load_groundtruth_depth_custom(png_path):
    # Load as grayscale / "I" to keep the integer values
    gt = Image.open(png_path).convert("I")
    gt = np.array(gt, dtype=np.float32)
    # Convert SUN dataset GT from decimillimeters to meters
    gt = gt / 1000.0
    # Mask out invalid pixels (e.g. zeros)
    mask = (gt > 1e-6)
    return gt, mask

def resize_depth_map(depth, target_shape):
    depth_img = Image.fromarray(depth)
    depth_resized = depth_img.resize((target_shape[1], target_shape[0]), Image.BICUBIC)
    return np.array(depth_resized, dtype=np.float32)


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

