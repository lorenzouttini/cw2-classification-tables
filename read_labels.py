import os
import pickle
import matplotlib.pyplot as plt

# Set the base paths
# base_path = "/home/cerendnr/object_detection/Code/data/CW2-Dataset/data/mit_76_459/76-459b/"
base_path = os.path.join("data", "mit_76_459", "76-459b")
label_file_path = os.path.join(base_path, "labels", "tabletop_labels.dat")
img_dir = os.path.join(base_path, "image")

# Load the pickle file containing the 3D list of polygon labels
with open(label_file_path, 'rb') as f:
    tabletop_labels = pickle.load(f)

# Get a sorted list of image filenames to ensure correct ordering
img_list = sorted(os.listdir(img_dir))

for polygon_list, img_name in zip(tabletop_labels, img_list):
    img_path = os.path.join(img_dir, img_name)
    img = plt.imread(img_path)
    plt.imshow(img)
    
    for polygon in polygon_list:
        # Each polygon is expected to be a two-element list: [list_of_xs, list_of_ys]
        xs = polygon[0] + [polygon[0][0]]  # Append the first x to close the polygon
        ys = polygon[1] + [polygon[1][0]]  # Append the first y to close the polygon
        plt.plot(xs, ys, 'r', linewidth=2)
    
    plt.axis('off')
    plt.show()
