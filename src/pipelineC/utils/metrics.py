import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def calc_iou(pred, target, n_classes=2):
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        pred: Prediction tensor (B, N) or (N,)
        target: Target tensor (B, N) or (N,)
        n_classes: Number of classes
        
    Returns:
        IoU for each class
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate IoU for each class
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # Avoid division by zero
        else:
            ious.append((intersection / union).item())
            
    return ious

def calc_accuracy(pred, target):
    """
    Calculate accuracy
    
    Args:
        pred: Prediction tensor (B, N) or (N,)
        target: Target tensor (B, N) or (N,)
        
    Returns:
        Accuracy value
    """
    pred = pred.view(-1)
    target = target.view(-1)
    correct = (pred == target).sum().float()
    total = pred.numel()
    return (correct / total).item()

def evaluate_segmentation(model, data_loader, device, criterion=None):
    """
    Evaluate segmentation model
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            pointclouds = batch["pointcloud"].to(device)
            labels = batch["point_labels"].to(device)
            
            # Forward pass
            outputs = model(pointclouds)
            pred = outputs.max(dim=2)[1]

            # Calculate loss if criterion provided
            if criterion is not None:
                outputs_flat = outputs.view(-1, outputs.size(2))  # (B*N, num_classes)
                labels_flat = labels.view(-1)                     # (B*N)
                loss = criterion(outputs_flat, labels_flat)
                total_loss += loss.item() * pointclouds.size(0)
            
            # Store predictions and targets
            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())
    
    # Calculate average loss
    if criterion is not None:
        avg_loss = total_loss / len(data_loader.dataset)
    else:
        avg_loss = 0.0

    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    iou_values = calc_iou(all_preds, all_targets)
    accuracy = calc_accuracy(all_preds, all_targets)
    
    # Calculate confusion matrix
    preds_np = all_preds.view(-1).numpy()
    targets_np = all_targets.view(-1).numpy()
    cm = confusion_matrix(targets_np, preds_np)
    
    # Calculate precision and recall for table class (class 1)
    if cm.shape[0] > 1 and cm.shape[1] > 1:
        precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = 0
        recall = 0
        f1 = 0
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "iou_background": iou_values[0],
        "iou_table": iou_values[1],
        "mean_iou": np.nanmean(iou_values),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }
    
    return metrics

def print_metrics(metrics):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
    """
    print("Segmentation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  IoU (Background): {metrics['iou_background']:.4f}")
    print(f"  IoU (Table): {metrics['iou_table']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])