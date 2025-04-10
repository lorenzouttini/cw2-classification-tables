import os
import torch
from torch.utils.data import DataLoader
from dataloader_custom import process_sequences
import cv2
import matplotlib.pyplot as plt
from model import DGCNNClassifier

# ------------------------
# Evaluation Function (For prediction only)
# ------------------------

def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    file_paths = []
    with torch.no_grad():
        for batch in dataloader:
            pointclouds = batch["pointcloud"].to(device)
            outputs = model(pointclouds)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            file_paths.extend(batch["file_path"])
    
    return all_preds, file_paths

# ------------------------
# Main Evaluation and Visualization Script
# ------------------------

def main():

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up a directory to reach project root
    base_path = os.path.normpath(os.path.join(script_dir, "../../data/RealSense/"))

    # Hyperparameters (must match your training configuration)
    batch_size = 16
    k_val = 20
    emb_dims = 1024
    dropout = 0.5
    num_classes = 2
    model_path = os.path.join(script_dir, "best_models/best_dgcnn_model.pth") # Default Path to the saved model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create visualization directory
    visualization_dir = os.path.join(script_dir, "visualize_custom_dataset_predictions")
    os.makedirs(visualization_dir, exist_ok=True)

    ucl_data_sequences = [
        "UCL_Data1"
        # "UCL_Data_2",
        # "UCL_Data_3"
    ]

    print("\nProcessing Test Sequences:")
    try:
        test_dataset = process_sequences(ucl_data_sequences, base_path, set_name="Test")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    except ValueError as e:
        print(f"Error processing sequences: {e}")
        # Try with a single sequence to debug
        print("Attempting to debug with first sequence only...")
        try:
            test_dataset = process_sequences([ucl_data_sequences[0]], base_path, set_name="Test")
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        except Exception as e:
            print(f"Still encountering error: {e}")
            raise
    
    # Load the trained model
    print(f"\nLoading best model from {model_path}")
    model = DGCNNClassifier(k=k_val, emb_dims=emb_dims, num_classes=num_classes, dropout=dropout)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Get predictions for the test set
    print("\nRunning predictions on test data...")
    predictions, file_paths = predict(model, test_loader, device)
    
    # Count predictions by class
    class_0_count = sum(1 for p in predictions if p == 0)
    class_1_count = sum(1 for p in predictions if p == 1)
    print(f"\nPrediction Summary:")
    print(f"- Class 0 (No Table): {class_0_count}")
    print(f"- Class 1 (Table): {class_1_count}")
    print(f"- Total: {len(predictions)}")
    
    # -------------------------------------------------------------------
    # Visualization: For each test sample, load the corresponding raw image
    # and overlay prediction
    # -------------------------------------------------------------------
    print("\nVisualizing predictions on raw test images...")
    print(f"Total test samples: {len(test_dataset)}")
    vis_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    processed_count = 0
    
    with torch.no_grad():
        for batch in vis_loader:
            pointcloud = batch["pointcloud"].to(device)
            depth_file_path = batch["file_path"][0]
            
            # Get prediction
            output = model(pointcloud)
            _, pred = torch.max(output, 1)
            predicted_label = pred.item()
            
            # Create a friendly prediction text
            pred_text = "Table Detected" if predicted_label == 1 else "No Table"

            # Load the raw image
            raw_image = cv2.imread(depth_file_path, cv2.IMREAD_COLOR)
            if raw_image is None:
                print(f"Could not load image: {depth_file_path}")
                continue

            # Overlay prediction on the image
            text = f"Prediction: {pred_text}"
            # Set color based on prediction (green for table, red for no table)
            color = (0, 255, 0) if predicted_label == 1 else (0, 0, 255)
            cv2.putText(raw_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)

            # Convert BGR to RGB for display using matplotlib
            image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(6, 4))
            plt.imshow(image_rgb)
            plt.title(text)
            
            # Extract sequence name and file name for unique saving
            sequence_name = os.path.basename(os.path.dirname(os.path.dirname(depth_file_path)))
            file_name = os.path.basename(depth_file_path)
            base_name = os.path.splitext(file_name)[0]
            
            save_path = os.path.join(visualization_dir, f"pred_{predicted_label}_{sequence_name}_{base_name}.png")
            plt.savefig(save_path)
            plt.close()
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(test_dataset)} images")
            
    print(f"Successfully processed and saved {processed_count} images to {visualization_dir}/")

if __name__ == "__main__":
    main()
