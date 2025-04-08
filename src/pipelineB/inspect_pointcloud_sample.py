import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import open3d as o3d
import matplotlib.path as mpath
from collections import Counter

### ----------- HELPER FUNCTIONS ----------- ###

def visualize_pointcloud_interactive(points, save_path="figures/preview.ply"):
    """
    Show a 3D point cloud using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # o3d.visualization.draw_geometries([pcd], window_name="Interactive Point Cloud",
    #                                   width=800, height=600)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved point cloud to: {save_path}")

def load_intrinsics(path):
    """
    Loads a 3x3 camera intrinsics matrix from intrinsics.txt.
    Format expected:
        fx 0  cx
        0  fy cy
        0  0   1
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    mat = [list(map(float, line.strip().split())) for line in lines]
    return np.array(mat)

def filter_points_in_polygon(points, intrinsics, polygon, image_shape):
    """
    Projects 3D points to image plane using intrinsics and filters
    only those that fall inside the polygon (2D).
    """

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Project points to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    # Form 2D point list and clamp to image boundaries
    uv = np.stack([u, v], axis=1)
    h, w = image_shape
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[valid]
    points = points[valid]

    # Create polygon path
    path = mpath.Path(np.stack([polygon[0], polygon[1]], axis=1))
    inside = path.contains_points(uv)

    return points[inside]


### ----------- CHECK IMBALANCE  ----------- ###

def check_dataset_balance(data_dir):
    labels = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".npy"):
            path = os.path.join(data_dir, fname)
            sample = np.load(path, allow_pickle=True).item()
            labels.append(sample["label"])

    count = Counter(labels)
    print(f"Total samples: {len(labels)}")
    print(f"Label counts: {count}")
    print(f"Class 0 (No Table): {count.get(0, 0)}")
    print(f"Class 1 (Table): {count.get(1, 0)}")

    # Optional: check balance ratio
    total = sum(count.values())
    print(f"Class 0 ratio: {count.get(0, 0) / total:.2%}")
    print(f"Class 1 ratio: {count.get(1, 0) / total:.2%}")


### ----------- VISUALIZE POINTCLOUDS ----------- ###

def inspect_sample(npy_path, scenes_root="data"):
    sample = np.load(npy_path, allow_pickle=True).item()
    points = sample["points"]
    label = sample["label"]

    # Infer scene name and frame index from filename
    basename = os.path.basename(npy_path)
    scene_name = "_".join(basename.split("_")[:-1])
    frame_idx = int(basename.split("_")[-1].split(".")[0])

    # Locate corresponding scene folder (search in data/)
    scene_path = None
    for root, dirs, files in os.walk(scenes_root):
        if scene_name in root and "image" in dirs:
            scene_path = root
            break
    if scene_path is None:
        print(f"Could not locate original scene folder for {scene_name}")
        return

    # Load original image
    image_dir = os.path.join(scene_path, "image")
    img_files = sorted(os.listdir(image_dir))
    img_path = os.path.join(image_dir, img_files[frame_idx])
    image = Image.open(img_path)

    # Load polygon labels
    label_file = os.path.join(scene_path, "labels", "tabletop_labels.dat")
    with open(label_file, 'rb') as f:
        label_data = pickle.load(f)
    polygons = label_data[frame_idx]

    # ---- RGB + Polygon Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(image)
    for polygon in polygons:
        xs = polygon[0] + [polygon[0][0]]
        ys = polygon[1] + [polygon[1][0]]
        ax1.plot(xs, ys, 'r', linewidth=2)
    ax1.set_title(f"{basename} | Label: {'Table' if label else 'No Table'}\nPoints: {len(points)}")
    ax1.axis('off')

    # ---- Point Cloud Plot ----
    # Only one polygon assumed
    intrinsic_path = os.path.join(scene_path, "intrinsics.txt")
    intrinsics = load_intrinsics(intrinsic_path)
    if len(polygons) > 0:
        table_points = np.empty((0, 3), dtype=np.float32)
        for i, polygon in enumerate(polygons):
            print(f"\nPolygon {i}: X={len(polygon[0])}, Y={len(polygon[1])}")
            mask_points = filter_points_in_polygon(points, intrinsics, polygon, image.size[::-1])
            table_points = np.vstack((table_points, mask_points))

        # polygon = polygons[0]  # Use the first polygon
        # table_points = filter_points_in_polygon(points, intrinsics, polygon, image.size[::-1])  # (H, W)

        if table_points.shape[0] > 0:
            ax2 = fig.add_subplot(122, projection='3d')
            p = table_points[::1]
            ax2.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=p[:, 2], cmap='plasma')
            ax2.set_title("Table Point Cloud Only")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
            ax2.view_init(elev=10, azim=135)
        else:
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title("No Points in Polygon")
            ax2.set_axis_off()
    else:
        print(" No polygons in this sample.")

    

    # Save the figure
    plt.tight_layout()
    os.makedirs("figures/point_clouds", exist_ok=True)
    fig_path = os.path.join("figures/point_clouds", f"{basename.replace('.npy', '.png')}")
    plt.savefig(fig_path)
    plt.close()

    # Print some points
    print(f"\n🔎 Point cloud preview (first 5 points):")
    for p in points[:1]:
        print(f"  ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # Save 3D point cloud as .ply
    ply_path = os.path.join("figures/point_clouds", f"{basename.replace('.npy', '.ply')}")
    visualize_pointcloud_interactive(points, save_path=ply_path)
    print(f"✅ {len(table_points)} table points retained after filtering.")
    # visualize_pointcloud_interactive(points)
    

# Entry point remains the same
if __name__ == "__main__":
    folder_path = "data/pointclouds"
    check_dataset_balance(folder_path)
    all_samples = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    for fname in all_samples[:3]:
        sample_path = os.path.join(folder_path, fname)
        inspect_sample(sample_path)
