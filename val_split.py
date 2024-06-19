import os
import random
import shutil

# Define the paths
images_train_path = 'images/train'
labels_train_path = 'labels/train'
images_val_path = 'images/val'
labels_val_path = 'labels/val'

# Create val directories if they don't exist
os.makedirs(images_val_path, exist_ok=True)
os.makedirs(labels_val_path, exist_ok=True)

# Get list of all image files in the train directory
image_files = [f for f in os.listdir(images_train_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Calculate 7% of the total number of images
val_count = int(len(image_files) * 0.07)

# Randomly select 7% of the images for validation
val_images = random.sample(image_files, val_count)

for image_file in val_images:
    # Move image file to val folder
    shutil.move(os.path.join(images_train_path, image_file), os.path.join(images_val_path, image_file))
    
    # Get the corresponding label file
    label_file = os.path.splitext(image_file)[0] + '.txt'
    
    # Move label file to val folder
    shutil.move(os.path.join(labels_train_path, label_file), os.path.join(labels_val_path, label_file))

print(f"Moved {val_count} image files and their corresponding label files to the val directory.")
