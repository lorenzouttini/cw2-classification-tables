import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataloader import process_sequences, compute_class_weights, random_augmentation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import DGCNNClassifier
import seaborn as sns


# ------------------------
# Training and Evaluation Functions
# ------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        pointclouds = batch["pointcloud"].to(device)  # (B, num_points, 3)
        labels = batch["label"].to(device)             # (B,)
        
        # Skip if batch size is 1 or less
        if pointclouds.size(0) <= 1:
            continue
            
        optimizer.zero_grad()
        outputs = model(pointclouds)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * pointclouds.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    
    if total == 0:
        return 0.0, 0.0
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            pointclouds = batch["pointcloud"].to(device)
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
# Main Training Script with 5-Fold Cross Validation and CosineAnnealingLR
# ------------------------

def main():
    # Hyperparameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.0001
    weight_decay = 1e-4  # L2 regularization
    k_val = 20
    emb_dims = 1024
    dropout = 0.5
    num_classes = 2
    n_splits = 5
    T_max = num_epochs  # Maximum number of epochs for cosine scheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two directories to reach project root
    base_path = os.path.normpath(os.path.join(script_dir, "../../data/CW2-Dataset/"))

    sequences_train = [
        "mit_32_d507/d507_2",
        "mit_76_459/76-459b",
        "mit_76_studyroom/76-1studyroom2",
        "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika",  # negative samples
        "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"
    ]
    sequences_test = [
        "harvard_c5/hv_c5_1",
        "harvard_c6/hv_c6_1",
        "harvard_c11/hv_c11_2",
        "harvard_tea_2/hv_tea2_2"  # Raw depth images are in "depth" folder here.
    ]
    
    print("Processing Training Sequences:")
    train_full_dataset = process_sequences(sequences_train, base_path, set_name="Train", transform=random_augmentation)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weights(train_full_dataset)
    class_weights = class_weights.to(device)
    print(f"Using class weights: {class_weights}")

    print("--------------------------------------------------------------------------")
    
    print("Processing Test Sequences:")
    test_dataset = process_sequences(sequences_test, base_path, set_name="Test", transform=None)
    
    # Create label list for StratifiedKFold
    labels = [train_full_dataset[i]["label"].item() for i in range(len(train_full_dataset))]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_val_accs = []
    fold_val_losses = []
    best_fold_model_state = None
    best_fold_val_acc = 0.0
    best_fold_val_loss = float('inf')
    all_fold_train_losses = []
    all_fold_val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\nStarting Fold {fold+1}/{n_splits}")
        print(f"Training size: {len(train_idx)} | Validation size: {len(val_idx)}")
        train_subset = Subset(train_full_dataset, train_idx)
        val_subset = Subset(train_full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        model = DGCNNClassifier(k=k_val, emb_dims=emb_dims, num_classes=num_classes, dropout=dropout)
        model = model.to(device)
        
        # Use weighted CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
        
        best_val_acc_fold = 0.0
        best_val_loss_fold = float('inf')
        train_losses_fold = []
        val_losses_fold = []
        train_accs_fold = []
        val_accs_fold = []
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            train_losses_fold.append(train_loss)
            val_losses_fold.append(val_loss)
            train_accs_fold.append(train_acc)
            val_accs_fold.append(val_acc)
            
            scheduler.step()
            
            print(f"Fold {fold+1} Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Updated best model selection criteria:
            if val_acc > best_val_acc_fold:
                best_val_acc_fold = val_acc
                best_val_loss_fold = val_loss
                best_model_state_fold = model.state_dict()
            elif val_acc == best_val_acc_fold and val_loss < best_val_loss_fold:
                best_val_loss_fold = val_loss
                best_model_state_fold = model.state_dict()
        
        print(f"Fold {fold+1} Best Val Acc: {best_val_acc_fold:.4f} with Loss: {best_val_loss_fold:.4f}")
        fold_val_accs.append(best_val_acc_fold)
        fold_val_losses.append(best_val_loss_fold)
        all_fold_train_losses.append(train_losses_fold)
        all_fold_val_losses.append(val_losses_fold)
        
        if best_val_acc_fold > best_fold_val_acc:
            best_fold_val_acc = best_val_acc_fold
            best_fold_val_loss = best_val_loss_fold
            best_fold_model_state = best_model_state_fold
        elif best_val_acc_fold == best_fold_val_acc and best_val_loss_fold < best_fold_val_loss:
            best_fold_val_loss = best_val_loss_fold
            best_fold_model_state = best_model_state_fold

        # Create figures directory if it doesn't exist
        figures_dir = os.path.join(script_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot loss curves for current fold
        epochs_arr = np.arange(1, num_epochs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_arr, train_losses_fold, label="Train Loss")
        plt.plot(epochs_arr, val_losses_fold, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold+1} Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"fold_{fold+1}_loss_curves.png"))
        plt.close()
        
        # Plot accuracy curves for current fold
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_arr, train_accs_fold, label="Train Accuracy", color='green')
        plt.plot(epochs_arr, val_accs_fold, label="Val Accuracy", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Fold {fold+1} Accuracy Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"fold_{fold+1}_accuracy_curves.png"))
        plt.close()
    
    # After cross-validation, determine best fold using our criterion:
    # Use tuple (accuracy, -loss) so that higher accuracy and lower loss yield a higher value.
    scores = [(acc, -loss) for acc, loss in zip(fold_val_accs, fold_val_losses)]
    best_fold_index = int(np.argmax(scores))
    best_fold_val_acc = fold_val_accs[best_fold_index]
    best_fold_val_loss = fold_val_losses[best_fold_index]
    
    print("\nCross Validation Complete.")
    print(f"Average Validation Accuracy over {n_splits} folds: {np.mean(fold_val_accs):.4f}")
    print(f"Best Fold {best_fold_index + 1} Validation Accuracy: {best_fold_val_acc:.4f} with Loss: {best_fold_val_loss:.4f}")

    
    plt.figure()
    plt.bar(np.arange(1, n_splits+1), fold_val_losses, tick_label=np.arange(1, n_splits+1))
    plt.xlabel("Fold")
    plt.ylabel("Best Validation Loss")
    plt.title("Best Validation Loss per Fold (Pipeline A)")
    plt.grid(True)
    plt.show()
    
    # Create best_models directory if it doesn't exist
    best_models_dir = os.path.join(script_dir, "best_models")
    os.makedirs(best_models_dir, exist_ok=True)
    
    model_save_path = os.path.join(best_models_dir, "best_dgcnn_model.pth")
    torch.save(best_fold_model_state, model_save_path)
    print(f"Best model saved to '{model_save_path}'")

    # Plot overall training and validation losses across folds
    plt.figure(figsize=(12, 6))
    epochs_arr = np.arange(1, num_epochs + 1)
    
    plt.subplot(1, 2, 1)
    for fold in range(n_splits):
        plt.plot(epochs_arr, all_fold_train_losses[fold], label=f"Fold {fold+1} Train")
        plt.plot(epochs_arr, all_fold_val_losses[fold], label=f"Fold {fold+1} Val", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Across Folds")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(1, n_splits+1), fold_val_accs, tick_label=np.arange(1, n_splits+1))
    plt.xlabel("Fold")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Best Validation Accuracy per Fold")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cross_validation_summary.png"))
    plt.close()
    
    # Evaluate best model on test set
    model = DGCNNClassifier(k=k_val, emb_dims=emb_dims, num_classes=num_classes, dropout=dropout)
    model.load_state_dict(torch.load(model_save_path))
    model = model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}")
    
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
    
    # Plot test performance metrics
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

if __name__ == "__main__":
    main()
