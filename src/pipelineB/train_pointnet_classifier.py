import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from pointnet_model import PointNetPlusPlus
from pointnet_dataset import PointCloudTableDataset

# Helper function to plot training curves
def plot_training_curves(train_vals, val_vals, ylabel, title, filename, batch_size):
    os.makedirs(f"figures/results/batch_{batch_size}", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_vals, label="Train")
    plt.plot(val_vals, label="Val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(f"figures/results/batch_{batch_size}", filename))
    plt.close()

def train_pointnet_classifier(
    data_dir,
    batch_size=16,
    epochs=50,
    lr=1e-3,
    num_points=1024,
    val_split=0.2,
    save_path="pointnet_table_classifier.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset 
    dataset = PointCloudTableDataset(data_dir, num_points=num_points)
    val_size = int(len(dataset) * val_split)                            # 20% for validation
    train_size = len(dataset) - val_size                                # 80% for training
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model =      PointNetPlusPlus(num_classes=2).to(device)      # 2 classes: table or no table
    optimizer =  optim.Adam(model.parameters(), lr=lr)           # Adam optimizer
    criterion =  nn.CrossEntropyLoss()

    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_recalls = []
    val_recalls = []
    train_precisions = []
    val_precisions = []

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0

        # Initialize counts for precision and recall
        train_tp, train_fp, train_fn = 0, 0, 0

        # Loop over training batches
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(points)   # Forward pass
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            train_correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Accumulate true positives, false positives, false negatives for binary class 1
            train_tp += ((pred == 1) & (labels == 1)).sum().item()
            train_fp += ((pred == 1) & (labels == 0)).sum().item()
            train_fn += ((pred == 0) & (labels == 1)).sum().item()

        # Compute epoch-level metrics for training
        train_acc = train_correct / total
        train_loss_epoch = train_loss / len(train_loader)
        train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0.0
        train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0.0

        # Append training metrics
        train_losses.append(train_loss_epoch)
        train_accuracies.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)

        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        val_tp, val_fp, val_fn = 0, 0, 0
        total_val = 0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                out = model(points)
                loss = criterion(out, labels)
                val_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                val_correct += (pred == labels).sum().item()
                total_val += labels.size(0)
                
                val_tp += ((pred == 1) & (labels == 1)).sum().item()
                val_fp += ((pred == 1) & (labels == 0)).sum().item()
                val_fn += ((pred == 0) & (labels == 1)).sum().item()

        val_acc = val_correct / len(val_ds)
        val_loss_epoch = val_loss / len(val_loader)
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0

        # Append validation metrics
        val_losses.append(val_loss_epoch)
        val_accuracies.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss_epoch:.4f}, "
              f"Train Acc: {train_acc:.2%}, "
              f"Train Prec: {train_precision:.2%}, "
              f"Train Recall: {train_recall:.2%}, "
              f"Val Loss: {val_loss_epoch:.4f}, "
              f"Val Acc: {val_acc:.2%}, "
              f"Val Prec: {val_precision:.2%}, "
              f"Val Recall: {val_recall:.2%}")

        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model to {save_path}")

    # Plot all training curves using the helper function
    plot_training_curves(train_accuracies, val_accuracies, "Accuracy", "Accuracy over Epochs", "accuracy_plot.png", batch_size)
    plot_training_curves(train_losses, val_losses, "Loss", "Loss over Epochs", "loss_plot.png", batch_size)
    plot_training_curves(train_precisions, val_precisions, "Precision", "Precision over Epochs", "precision_plot.png", batch_size)
    plot_training_curves(train_recalls, val_recalls, "Recall", "Recall over Epochs", "recall_plot.png", batch_size)

    print(f"Training complete. Best Val Acc: {best_acc:.2%}")

if __name__ == "__main__":
    train_pointnet_classifier(data_dir="data/pointclouds")

