import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

scene_path = "data/ucl_data_20250408_143718/ucl1"
image_dir = os.path.join(scene_path, "image")
gt_dir = os.path.join(scene_path, "depth")
pred_dir = os.path.join(scene_path, "depth_pred")

image_files = sorted(os.listdir(image_dir))
gt_files = sorted(os.listdir(gt_dir))
pred_files = sorted(os.listdir(pred_dir))

print(f"Images:      {len(image_files)} files")
print(f"GT Depths:   {len(gt_files)} files")
print(f"Pred Depths: {len(pred_files)} files\n")

# Explore first N files
N = 5
for i in range(N):
    img_name = image_files[i]
    gt_name = gt_files[i]
    pred_name = pred_files[i]

    print(f"\n--- Frame {i+1} ---")
    print(f"Image file:     {img_name}")
    print(f"GT depth file:  {gt_name}")
    print(f"Pred depth file:{pred_name}")

    # Load image
    img = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
    gt_depth = np.array(Image.open(os.path.join(gt_dir, gt_name)).convert("I"), dtype=np.float32) / 1000.0
    pred_depth = np.array(Image.open(os.path.join(pred_dir, pred_name)).convert("I"), dtype=np.float32) / 1000.0

    print(f"Image shape:     {np.array(img).shape}")
    print(f"GT depth shape:  {gt_depth.shape}, min: {gt_depth.min():.3f}, max: {gt_depth.max():.3f}")
    print(f"Pred depth shape:{pred_depth.shape}, min: {pred_depth.min():.3f}, max: {pred_depth.max():.3f}")

    # Mask stats
    valid_gt = gt_depth > 0.01
    valid_pred = pred_depth > 0.01
    print(f"Valid GT pixels:     {np.sum(valid_gt)} / {gt_depth.size}")
    print(f"Valid Pred pixels:   {np.sum(valid_pred)} / {pred_depth.size}")

    # Visual comparison
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("RGB Image")
    axs[1].imshow(gt_depth, cmap='plasma')
    axs[1].set_title("GT Depth")
    axs[2].imshow(pred_depth, cmap='plasma')
    axs[2].set_title("Predicted Depth")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs("figures/results/ucl_inspection", exist_ok=True)
    plt.savefig(f"figures/results/ucl_inspection/frame_{i+1}.png")
    plt.show()
