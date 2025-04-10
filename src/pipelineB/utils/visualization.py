
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


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