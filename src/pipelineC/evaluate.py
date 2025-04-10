import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelineC.models.dgcnn import DGCNN_Seg
from pipelineC.dataset import PointCloudSegmentationDataset, depth_to_pointcloud
from pipelineC.utils.metrics import evaluate_segmentation, print_metrics
from pipelineC.utils.visualization import visualize_prediction, visualize_with_open3d

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DGCNN model on Harvard dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--num_vis', type=int, default=3, help='Number of samples to visualize')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # CUDA setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = DGCNN_Seg(k=20, num_classes=2, emb_dims=1024)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Dataset paths
    base_path = "./data/CW2-Dataset/"
    
    # Harvard sequences
    harvard_sequences = [
        "harvard_c5/hv_c5_1",
        "harvard_c6/hv_c6_1",
        "harvard_c11/hv_c11_2",
        "harvard_tea_2/hv_tea2_2"
    ]

    # Directory to save visualizations
    vis_save_dir = "./src/pipelineC/results/visualizations"
    os.makedirs(vis_save_dir, exist_ok=True)
    
    # Evaluate on each Harvard sequence separately
    for seq in harvard_sequences:
        seq_path = os.path.join(base_path, seq)
        # Determine depth folder
        depth_folder = "depth" if "harvard_tea_2" in seq else "depthTSDF"
        
        # Create dataset
        dataset = PointCloudSegmentationDataset(
            root_dir=seq_path,
            depth_folder=depth_folder,
            annotation_path="labels/tabletop_labels.dat",
            intrinsics_path=os.path.join(seq_path, "intrinsics.txt"),
            num_points=1024,
            transform=None,
            verbose=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"\nEvaluating on {seq}...")
        metrics = evaluate_segmentation(model, dataloader, device)
        print_metrics(metrics)
        
        # Visualize predictions if requested
        if args.visualize:
            print(f"\nVisualizing predictions for {seq}...")
            vis_indices = np.random.choice(len(dataset), min(args.num_vis, len(dataset)), replace=False)
            
            for i in vis_indices:
                # Get sample
                sample = dataset[i]
                filename = sample["filename"]
                pointcloud = sample["pointcloud"]
                labels = sample["point_labels"]
                
                # Load depth image for visualization
                depth_path = os.path.join(seq_path, depth_folder, filename)
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                
                # Get prediction
                with torch.no_grad():
                    input_pc = pointcloud.unsqueeze(0).to(device)
                    output = model(input_pc)
                    pred = output.max(dim=2)[1].squeeze(0)
                
                # Visualize
                fig = visualize_prediction(depth_img, pointcloud, labels, pred, filename)
                
                # Save the predictions
                save_path = os.path.join(vis_save_dir, f"{os.path.splitext(filename)[0]}_vis.png")
                fig.savefig(save_path)
                plt.close(fig)
                print(f"Saved visualization to {save_path}")
                
                # Optionally, visualize with Open3D for interactive inspection
                visualize_with_open3d(pointcloud, pred, f"{filename} (Prediction)")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()