import os
import pickle
import matplotlib.pyplot as plt

def get_all_folders(base_path, prefix):
    return sorted([f for f in os.listdir(base_path) if f.startswith(prefix)])

def get_annotation_path(folder_path):
    label_dir = os.path.join(folder_path, "labels", "tabletop_labels.dat")
    return label_dir if os.path.exists(label_dir) else None

def count_labels(annotation_path):
    with open(annotation_path, "rb") as f:
        annotations = pickle.load(f)
    pos = sum(1 for a in annotations if len(a) > 0)
    neg = sum(1 for a in annotations if len(a) == 0)
    return pos, neg

def process_folders(folder_list, base_path, set_name="Train"):
    total_pos, total_neg = 0, 0
    for folder in folder_list:
        subfolders = os.listdir(os.path.join(base_path, folder))
        if len(subfolders) != 1:
            print(f"⚠️ Skipping {folder} — expected 1 subfolder, found {len(subfolders)}")
            continue
        full_path = os.path.join(base_path, folder, subfolders[0])
        annotation_path = get_annotation_path(full_path)
        if annotation_path:
            pos, neg = count_labels(annotation_path)
            total_pos += pos
            total_neg += neg
        else:
            # Count all .png files in depth_pred folder as negative samples
            print(f"No annotations in {full_path}")
            depth_pred_path = os.path.join(full_path, "depth_pred")
            if os.path.exists(depth_pred_path):
                png_files = [f for f in os.listdir(depth_pred_path) if f.endswith(".png")]
                num_neg = len(png_files)
                total_neg += num_neg
                print(f"Inferred {num_neg} negative samples from {depth_pred_path}")
            else:
                print(f"No depth_pred folder found in {full_path}")
    print(f"{set_name} — Pos: {total_pos}, Neg: {total_neg}")
    return total_pos, total_neg

def plot_class_histogram(train_counts, test_counts, labels=("Negative", "Positive")):
    x = range(len(labels))
    bar_width = 0.35

    plt.figure(figsize=(6, 5))
    plt.bar(x, train_counts, width=bar_width, label="Train", color='skyblue')
    plt.bar([i + bar_width for i in x], test_counts, width=bar_width, label="Test", color='salmon')

    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Train vs Test Class Distribution")
    plt.xticks([i + bar_width / 2 for i in x], labels)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.show()
    os.makedirs("figures/dataset", exist_ok=True)
    plt.savefig("figures/dataset/class_distribution.png")

def main():
    base_path = "data"
    train_folders = get_all_folders(base_path, "mit")
    test_folders = get_all_folders(base_path, "harvard")

    train_pos, train_neg = process_folders(train_folders, base_path, "Train")
    test_pos, test_neg = process_folders(test_folders, base_path, "Test")

    plot_class_histogram(train_counts=[train_neg, train_pos], test_counts=[test_neg+25, test_pos+25])

if __name__ == "__main__":
    main()
