import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pointnet_model import PointNetPlusPlus
from pointnet_dataset import PointCloudTableDataset

def evaluate_model(model_path, data_dir, batch_size=16, num_points=1024, save_plot=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset =       PointCloudTableDataset(data_dir, num_points=num_points)
    dataloader =    DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model =     PointNetPlusPlus(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    tp, fp, fn = 0, 0, 0
    misclassified = []

    with torch.no_grad():
        for i, (points, labels) in enumerate(dataloader):
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

            for j in range(len(preds)):
                if preds[j] != labels[j]:
                    misclassified.append((i * batch_size + j, preds[j].item(), labels[j].item()))

    # Metrics
    acc = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\n Evaluation Results:")
    print(f"Accuracy : {acc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"Misclassified samples: {len(misclassified)} / {total}")
    for idx, pred, true in misclassified[:10]:
        print(f"  Sample {idx}: Predicted {pred}, True {true}")

    # Optional: Save as bar chart
    if save_plot:
        os.makedirs("figures/results/test", exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.bar(["Accuracy", "Precision", "Recall"], [acc, precision, recall], color=["skyblue", "lightgreen", "salmon"])
        plt.ylim(0, 1)
        plt.title("Evaluation Metrics on Test Set")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig("figures/results/test/eval_metrics.png")
        plt.close()
        print("Saved evaluation plot to: figures/results/eval/eval_metrics.png")

if __name__ == "__main__":
    evaluate_model(
        model_path="pointnet_table_classifier.pth",
        data_dir="data/pointclouds/test",  # Folder with .npy files from Harvard
        batch_size=16
    )

