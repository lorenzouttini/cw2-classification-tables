import os
import torch
from torch.utils.data import DataLoader
from pointnet_model import PointNetPlusPlus
from pointnet_dataset import PointCloudTableDataset

def evaluate_model(model_path, data_dir, batch_size=16, num_points=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = PointCloudTableDataset(data_dir, num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = PointNetPlusPlus(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    misclassified = []

    with torch.no_grad():
        for i, (points, labels) in enumerate(dataloader):
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Save misclassified indices
            for j in range(len(preds)):
                if preds[j] != labels[j]:
                    misclassified.append((i * batch_size + j, preds[j].item(), labels[j].item()))

    acc = correct / total
    print(f"Evaluation Accuracy: {acc:.2%}")
    print(f"Misclassified samples: {len(misclassified)} / {total}")
    for idx, pred, true in misclassified[:10]:
        print(f"  Sample {idx}: Predicted {pred}, True {true}")

if __name__ == "__main__":
    evaluate_model(
        model_path="pointnet_table_classifier.pth",
        data_dir="data/pointclouds_test",  # <- folder with npy files from Harvard or RealSense
        batch_size=16
    )
