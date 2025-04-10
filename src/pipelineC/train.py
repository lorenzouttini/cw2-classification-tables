import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import sys
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelineC.models.dgcnn import DGCNN_Seg
from pipelineC.dataset import create_datasets, create_train_val_split, random_augmentation
from pipelineC.utils.metrics import evaluate_segmentation, print_metrics
from pipelineC.utils.visualization import plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(description='Train DGCNN for point cloud segmentation')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points in each point cloud')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Fraction of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--k', type=int, default=20, help='Number of nearest neighbors')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of embeddings')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--table_weight', type=float, default=2.0, 
                        help='Weight for table class in loss function (to handle class imbalance)')
    parser.add_argument('--save_dir', type=str, default='src/pipelineC/checkpoints', help='Directory to save checkpoints')
    
    return parser.parse_args()

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create a progress bar for this epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                leave=True, ncols=100)
    
    for i, batch in enumerate(pbar):
        pointclouds = batch["pointcloud"].to(device)  # (B, N, 3)
        labels = batch["point_labels"].to(device)     # (B, N)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(pointclouds)  # (B, N, num_classes)
        
        # Reshape for loss calculation
        outputs = outputs.view(-1, 2)  # (B*N, 2)
        labels = labels.view(-1)       # (B*N)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * pointclouds.size(0)
        pred = outputs.max(dim=1)[1]
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def main():
    args = parse_args()
    
    # CUDA setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Dataset paths
    base_path = "./data/CW2-Dataset/"
    
    # Training sequences
    sequences_train = [
        "mit_32_d507/d507_2",
        "mit_76_459/76-459b",
        "mit_76_studyroom/76-1studyroom2",
        "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika",
        "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"
    ]
    
    # Testing sequences (Harvard data)
    sequences_test = [
        "harvard_c5/hv_c5_1",
        "harvard_c6/hv_c6_1",
        "harvard_c11/hv_c11_2",
        "harvard_tea_2/hv_tea2_2",
    ]
    
    # Create datasets
    print("Creating datasets...")
    mit_dataset, harvard_dataset = create_datasets(
        base_path, 
        sequences_train, 
        sequences_test, 
        transform=random_augmentation
    )

    # Split MIT data into train and validation
    train_dataset, val_dataset = create_train_val_split(mit_dataset, val_ratio=args.val_ratio, seed=42)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )

    val_loader = DataLoader( 
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )
    
    test_loader = DataLoader(
        harvard_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(harvard_dataset)}")
    
    # Create model
    model = DGCNN_Seg(
        k=args.k,
        num_classes=2,
        emb_dims=args.emb_dims,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Loss function with class weighting to handle imbalance
    class_weights = torch.FloatTensor([1.0, args.table_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    
    # Training loop
    print("Starting training...")
    best_val_iou = 0.0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_ious = []
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate on validation set
        metrics = evaluate_segmentation(model, val_loader, device, criterion)
        val_losses.append(metrics["loss"])
        val_ious.append(metrics["iou_table"])
        val_accs.append(metrics["accuracy"])
        
        # Print training and validation metrics
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {metrics['loss']:.4f}")  # Print validation loss
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Validation IoU (Table): {metrics['iou_table']:.4f}")
        
        # Save model if validation IoU improves
        if metrics["iou_table"] > best_val_iou:
            best_val_iou = metrics["iou_table"]
            model_path = os.path.join(args.save_dir, f"dgcnn_seg_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with Validation Table IoU: {best_val_iou:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(args.save_dir, f"dgcnn_seg_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    model_path = os.path.join(args.save_dir, f"dgcnn_seg_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model after {args.epochs} epochs")
    
    # Final evaluation on Harvard test set
    print("\nPerforming final evaluation on Harvard test set...")
    test_metrics = evaluate_segmentation(model, test_loader, device)
    print("\nTest Set Results (Harvard data):")
    print_metrics(test_metrics)

    # Save training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_ious, args.save_dir)    
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()