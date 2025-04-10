import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from matplotlib.path import Path

from datetime import datetime
import os

os.environ['XDG_SESSION_TYPE'] = 'x11'

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_ious, save_path):
    """
    Plot and save training curves comparing training and validation metrics
    
    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        val_ious: List of validation IoUs per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Directory to save plots
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training curves
    plt.figure(figsize=(8, 6))
    
    # Plot loss and metrics on separate subplots
    # 1. Loss curve
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, '-', label='Training Loss')
    plt.plot(epochs, val_losses, '-', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    
    # 2. Accuracy and IoU
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.plot(epochs, val_ious, 'm-', marker='^', label='Validation IoU (Table)')
    plt.title('Accuracy and IoU Metrics', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save figures
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    plt.savefig(os.path.join(save_path, f'training_curves_{timestamp}.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(save_path, f'training_curves_{timestamp}.pdf'), bbox_inches='tight')
    
    # Save data as text file for future reference
    metrics_data = "epoch,train_loss,train_accuracy,val_loss,val_accuracy,val_iou\n"
    for i, epoch in enumerate(epochs):
        metrics_data += f"{epoch},{train_losses[i]},{train_accs[i]},{val_losses[i]},{val_accs[i]},{val_ious[i]}\n"
    
    # Write to text file
    with open(os.path.join(save_path, f'training_metrics_{timestamp}.txt'), 'w') as f:
        f.write(metrics_data)
    
    print(f"Training curves saved to {save_path}")
    
    # Show the plot
    plt.show()

def visualize_with_open3d(pointcloud, labels, filename, voxel_size=0.02):
    """Visualize a point cloud with segmentation labels using Open3D."""
    # Move tensors to CPU if they're on GPU
    pointcloud = pointcloud.cpu() if isinstance(pointcloud, torch.Tensor) else pointcloud
    labels = labels.cpu() if isinstance(labels, torch.Tensor) else labels
    
    # Convert to numpy arrays
    pointcloud = pointcloud.numpy() if isinstance(pointcloud, torch.Tensor) else pointcloud
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    
    # Set colors based on labels (red for table, blue for background)
    colors = np.zeros((len(pointcloud), 3))
    colors[labels == 0] = [0, 0, 1]  # Blue for background
    colors[labels == 1] = [1, 0, 0]  # Red for table
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Print statistics about the point cloud
    table_count = np.sum(labels == 1)
    total_count = len(labels)
    print(f"\nPoint cloud statistics for {filename}:")
    print(f"  Total points: {total_count:,}")
    print(f"  Table points: {table_count:,} ({table_count/total_count*100:.1f}%)")
    print(f"  Background points: {total_count-table_count:,} ({(total_count-table_count)/total_count*100:.1f}%)")
    
    # Optional: Downsample for better performance if very large
    if len(pointcloud) > 100000:
        print(f"  Downsampling for visualization from {len(pointcloud):,} points...")
        # Use the provided voxel size for more effective downsampling
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  Downsampled to {len(pcd.points):,} points")
    
    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud: {filename}")

def visualize_prediction(depth_img, pointcloud, labels, pred_labels, filename, max_display_points=5000):
    """
    Visualize depth image with predicted segmentation
    
    Args:
        depth_img: Depth image
        pointcloud: Point cloud (N, 3)
        labels: Ground truth labels (N,)
        pred_labels: Predicted labels (N,)
        filename: Image filename
        max_display_points: Maximum number of points to display
    """
    # Move tensors to CPU if they're on GPU
    pointcloud = pointcloud.cpu() if isinstance(pointcloud, torch.Tensor) else pointcloud
    labels = labels.cpu() if isinstance(labels, torch.Tensor) else labels
    pred_labels = pred_labels.cpu() if isinstance(pred_labels, torch.Tensor) else pred_labels
    
    # Convert to numpy arrays
    pointcloud = pointcloud.numpy() if isinstance(pointcloud, torch.Tensor) else pointcloud
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    pred_labels = pred_labels.numpy() if isinstance(pred_labels, torch.Tensor) else pred_labels
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(18, 8))
    
    # 2D visualization - depth image (left subplot)
    ax1 = fig.add_subplot(131)
    depth_viz = ax1.imshow(depth_img, cmap='viridis')
    cbar = fig.colorbar(depth_viz, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (mm)', rotation=270, labelpad=15)
    ax1.set_title(f"Depth Image\n{filename}", fontsize=14)
    
    # Display depth stats
    valid_depth = depth_img[depth_img > 0]
    if len(valid_depth) > 0:
        min_depth = np.min(valid_depth)
        max_depth = np.max(valid_depth)
        mean_depth = np.mean(valid_depth)
        ax1.text(10, 20, f"Min: {min_depth:.1f}, Max: {max_depth:.1f}, Mean: {mean_depth:.1f}", 
                 color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.5))
    
    # 3D visualization - ground truth (middle subplot)
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Downsample for display if needed
    if len(pointcloud) > max_display_points:
        indices = np.random.choice(len(pointcloud), max_display_points, replace=False)
        display_points = pointcloud[indices]
        display_labels = labels[indices]
    else:
        display_points = pointcloud
        display_labels = labels
    
    # Plot background and table points (ground truth)
    bg_points = display_points[display_labels == 0]
    table_points = display_points[display_labels == 1]
    
    if len(bg_points) > 0:
        ax2.scatter(
            bg_points[:, 0], bg_points[:, 1], bg_points[:, 2], 
            c='blue', s=1, alpha=0.5, label='Background (GT)'
        )
    
    if len(table_points) > 0:
        ax2.scatter(
            table_points[:, 0], table_points[:, 1], table_points[:, 2], 
            c='red', s=2, alpha=0.7, label='Table (GT)'
        )
    
    ax2.set_title(f"Ground Truth\n{filename}", fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # 3D visualization - prediction (right subplot)
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Downsample for display if needed (use same indices as ground truth)
    if len(pointcloud) > max_display_points:
        display_pred_labels = pred_labels[indices]
    else:
        display_pred_labels = pred_labels
    
    # Plot background and table points (prediction)
    bg_points_pred = display_points[display_pred_labels == 0]
    table_points_pred = display_points[display_pred_labels == 1]
    
    if len(bg_points_pred) > 0:
        ax3.scatter(
            bg_points_pred[:, 0], bg_points_pred[:, 1], bg_points_pred[:, 2], 
            c='blue', s=1, alpha=0.5, label='Background (Pred)'
        )
    
    if len(table_points_pred) > 0:
        ax3.scatter(
            table_points_pred[:, 0], table_points_pred[:, 1], table_points_pred[:, 2], 
            c='red', s=2, alpha=0.7, label='Table (Pred)'
        )
    
    ax3.set_title(f"Prediction\n{filename}", fontsize=14)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # Auto-scale all 3D plots to match
    for ax in [ax2, ax3]:
        max_range = np.max([
            np.max(display_points[:, 0]) - np.min(display_points[:, 0]),
            np.max(display_points[:, 1]) - np.min(display_points[:, 1]),
            np.max(display_points[:, 2]) - np.min(display_points[:, 2])
        ])
        mid_x = (np.max(display_points[:, 0]) + np.min(display_points[:, 0])) * 0.5
        mid_y = (np.max(display_points[:, 1]) + np.min(display_points[:, 1])) * 0.5
        mid_z = (np.max(display_points[:, 2]) + np.min(display_points[:, 2])) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

    return fig