import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import process_sequences
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from model import DGCNNClassifier

# ------------------------
# Evaluation/Test Function
# ------------------------

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            pointclouds = batch["pointcloud"].to(device)  # (B, num_points, 3)
            labels = batch["label"].to(device)
            
            outputs = model(pointclouds)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * pointclouds.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if total == 0:
        return 0.0, 0.0, None, None
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds

# ------------------------
# Visualization Function
# ------------------------

def visualize_sample(sample):
    """
    Use the file_path in the sample to read the depth image (raw image)
    and visualize the calculated point cloud side by side.
    """
    # Read depth image (grayscale)
    depth_img = cv2.imread(sample["file_path"], cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError("Depth image not found for visualization.")
    # Get the point cloud
    pc = sample["pointcloud"].numpy()

    fig = plt.figure(figsize=(12, 6))
    
    # Left: Depth image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(depth_img, cmap='gray')
    ax1.set_title("Depth Image")
    ax1.axis('off')
    
    # Right: 3D point cloud scatter plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=pc[:, 2], cmap='viridis')
    ax2.set_title("3D Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    plt.tight_layout()
    plt.show()

# ------------------------
# Main Program: Creating Test Datasets, Evaluation and Visualization
# ------------------------

def main():

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two directories to reach project root
    base_path = os.path.normpath(os.path.join(script_dir, "../../data/CW2-Dataset/"))

    # Hyperparameters (must match the values used during training)
    batch_size = 16
    k_val = 20
    emb_dims = 1024
    dropout = 0.5
    num_classes = 2

    # Default Path to the saved model
    model_path = os.path.join(script_dir, "best_models/best_dgcnn_model.pth")  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories for visualization
    os.makedirs(os.path.join(script_dir, "visualize_test_predictions"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "visualize_test_predictions/actual1_pred1"), exist_ok=True)  # True Positives
    os.makedirs(os.path.join(script_dir, "visualize_test_predictions/actual1_pred0"), exist_ok=True)  # False Negatives
    os.makedirs(os.path.join(script_dir, "visualize_test_predictions/actual0_pred1"), exist_ok=True)  # False Positives
    os.makedirs(os.path.join(script_dir, "visualize_test_predictions/actual0_pred0"), exist_ok=True)  # True Negatives
    
    # Test sequence definitions
    sequences_test = [
        "harvard_c5/hv_c5_1",
        "harvard_c6/hv_c6_1",
        "harvard_c11/hv_c11_2",
        "harvard_tea_2/hv_tea2_2"  # Raw depth images in "depth" folder
    ]

    print("\nProcessing Test Sequences:")
    test_dataset = process_sequences(sequences_test, base_path, set_name="Test", transform=None)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load the model
    print(f"\nLoading best model from {model_path}")
    model = DGCNNClassifier(k=k_val, emb_dims=emb_dims, num_classes=num_classes, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate on the test set
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    # Plot test performance metrics
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    metrics = ['Loss', 'Accuracy']
    values = [test_loss, test_acc]
    plt.bar(metrics, values, color=['#ff9999', '#66b3ff'])
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.ylim(0, max(values) + 0.1)
    plt.title('Test Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "test_performance.png"))
    plt.close()
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=["No Table (0)", "Table (1)"])
    print("\nConfusion Matrix (Actual ↓ | Predicted →):")
    print("               Predicted")
    print("             |   0   |   1")
    print("-------------+-------+------")
    print(f"Actual   0   |  {cm[0][0]:>3}  |  {cm[0][1]:>3}")
    print(f"Actual   1   |  {cm[1][0]:>3}  |  {cm[1][1]:>3}")
    print("\nClassification Report:")
    print(cr)
    
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["No Table", "Table"],
                yticklabels=["No Table", "Table"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"))
    plt.close()
    
    # -------------------------------------------------------------------
    # Visualization: Overlay model predictions and actual labels on raw test images and save.
    # -------------------------------------------------------------------
    print("\nVisualizing predictions on raw test images...")
    print(f"Total test samples: {len(test_dataset)}")
    vis_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    processed_count = 0
    
    # Counters for categories
    tp_count = 0  # Actual 1, Predicted 1
    fn_count = 0  # Actual 1, Predicted 0
    fp_count = 0  # Actual 0, Predicted 1
    tn_count = 0  # Actual 0, Predicted 0
    
    with torch.no_grad():
        for batch in vis_loader:
            pointcloud = batch["pointcloud"].to(device)  # (1, num_points, 3)
            label = batch["label"].item()
            depth_file_path = batch["file_path"][0]
            
            # print(f"Processing file {processed_count+1}: {depth_file_path}")
            
            # Get the model prediction
            output = model(pointcloud)
            _, pred = torch.max(output, 1)
            predicted_label = pred.item()

            # Read the raw (depth) image as color
            raw_image = cv2.imread(depth_file_path, cv2.IMREAD_COLOR)
            if raw_image is None:
                print(f"Could not load raw image: {depth_file_path}")
                continue
            
            # Add the prediction and actual label text on the image
            text = f"Pred: {predicted_label}, Actual: {label}"
            cv2.putText(raw_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Convert BGR to RGB for matplotlib display
            image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 4))
            plt.imshow(image_rgb)
            plt.title(text)
            
            # Extract sequence and file name information to save the file
            sequence_name = os.path.basename(os.path.dirname(os.path.dirname(depth_file_path)))
            file_name = os.path.basename(depth_file_path)
            base_name = os.path.splitext(file_name)[0]
            
            # Determine the folder to save based on the actual and predicted labels
            if label == 1 and predicted_label == 1:
                subdir = "actual1_pred1"
                tp_count += 1
            elif label == 1 and predicted_label == 0:
                subdir = "actual1_pred0"
                fn_count += 1
            elif label == 0 and predicted_label == 1:
                subdir = "actual0_pred1"
                fp_count += 1
            else:  # label == 0 and predicted_label == 0:
                subdir = "actual0_pred0"
                tn_count += 1
                
            save_path = os.path.join(script_dir, "visualize_test_predictions", subdir, f"prediction_{sequence_name}_{base_name}.png")
            plt.savefig(save_path)
            plt.close()
            processed_count += 1
            
    print(f"Processed and saved {processed_count} images:")
    print(f"- True Positives (actual1_pred1): {tp_count}")
    print(f"- False Negatives (actual1_pred0): {fn_count}")
    print(f"- False Positives (actual0_pred1): {fp_count}")
    print(f"- True Negatives (actual0_pred0): {tn_count}")

if __name__ == "__main__":
    main()
