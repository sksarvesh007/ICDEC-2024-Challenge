import os
import cv2
import numpy as np
from tqdm import tqdm

# Load YOLO annotations from a file
def load_yolo_annotations(file_path):
    with open(file_path, 'r') as f:
        annotations = [line.strip().split() for line in f.readlines()]
    return annotations

# Save YOLO annotations to a file
def save_yolo_annotations(file_path, annotations):
    with open(file_path, 'w') as f:
        for annotation in annotations:
            line = ' '.join([str(x) for x in annotation])
            f.write(line + '\n')

# Convert YOLO annotation to bounding box coordinates
def yolo_to_bbox(yolo_annotation, img_width, img_height):
    class_id, x_centre, y_centre, width, height = map(float, yolo_annotation)
    x_centre *= img_width
    y_centre *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_centre - width / 2)
    x2 = int(x_centre + width / 2)
    y1 = int(y_centre - height / 2)
    y2 = int(y_centre + height / 2)
    return int(class_id), x1, y1, x2, y2

# Convert bounding box coordinates to YOLO annotation
def bbox_to_yolo(bbox, img_width, img_height):
    class_id, x1, y1, x2, y2 = bbox
    x_centre = (x1 + x2) / 2
    y_centre = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    x_centre /= img_width
    y_centre /= img_height
    width /= img_width
    height /= img_height
    return [str(class_id), f'{x_centre:.6f}', f'{y_centre:.6f}', f'{width:.6f}', f'{height:.6f}']

def check_overlap(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    if x1_min >= x2_max or x1_max <= x2_min or y1_min >= y2_max or y1_max <= y2_min:
        return False
    return True

# Find the height of a matching class in the second image
def find_matching_class_height(bboxes, class_id):
    for bbox in bboxes:
        if bbox[0] == class_id:
            return (bbox[2] + bbox[3]) // 2
    return None

# Cut-paste augmentation function with poison blending
def cut_paste_augmentation(image1, bboxes1, image2, bboxes2, alpha=0.8):
    img1 = image1.copy()
    img2 = image2.copy()
    for bbox in bboxes1:
        class_id, x1, y1, x2, y2 = bbox
        patch = img1[y1:y2, x1:x2]

        # Find the height level of the same class object in image2
        matching_height = find_matching_class_height(bboxes2, class_id)
        if matching_height is not None:
            patch_height = y2 - y1
            new_y1 = max(0, matching_height - patch_height // 2)
            new_y2 = new_y1 + patch_height
            if new_y2 > img2.shape[0]:
                new_y1 -= (new_y2 - img2.shape[0])
                new_y2 = img2.shape[0]
        else:
            new_y1 = np.random.randint(0, img2.shape[0] - (y2 - y1))
            new_y2 = new_y1 + (y2 - y1)

        # Check if the patch width exceeds the available width in image2
        max_patch_width = img2.shape[1] - (x2 - x1)
        if x2 - x1 > max_patch_width:
            continue  # Skip this patch if it's too wide for image2

        # Try placing the patch at a random x position until it does not overlap
        max_attempts = 1000
        placed = False
        for _ in range(max_attempts):
            new_x1 = np.random.randint(0, img2.shape[1] - (x2 - x1))
            new_x2 = new_x1 + (x2 - x1)

            overlap = False
            for existing_bbox in bboxes2:
                if check_overlap((new_x1, new_y1, new_x2, new_y2), existing_bbox[1:]):
                    overlap = True
                    break

            if not overlap:
                # Blend the patch into the image
                img2[new_y1:new_y2, new_x1:new_x2] = cv2.addWeighted(
                    img2[new_y1:new_y2, new_x1:new_x2], 1 - alpha, patch, alpha, 0
                )
                bboxes2.append((class_id, new_x1, new_y1, new_x2, new_y2))
                placed = True
                break

        if not placed:
            tqdm.write(f"Failed to place patch for bbox {bbox} without overlap after {max_attempts} attempts.")

    return img2, bboxes2

# Apply cut-paste augmentation to sequentially consecutive image pairs
def apply_augmentation_to_dataset(dataset_root_dir, num_pairs=3000):
    images_dir = os.path.join(dataset_root_dir, 'images', 'train')
    labels_dir = os.path.join(dataset_root_dir, 'labels', 'train')
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.jpeg')]
    
    if len(image_files) < 2:
        print("Error: There are not enough images in the directory to perform augmentation.")
        return
    
    augmented_count = 0
    i = 0
    pbar = tqdm(total=num_pairs)
    while augmented_count < num_pairs and i < len(image_files) - 1:
        image1_path = os.path.join(images_dir, image_files[i])
        image2_path = os.path.join(images_dir, image_files[i + 1])
        
        annotations1_path = os.path.join(labels_dir, image_files[i].replace('.jpg', '.txt').replace('.jpeg', '.txt'))
        annotations2_path = os.path.join(labels_dir, image_files[i + 1].replace('.jpg', '.txt').replace('.jpeg', '.txt'))
        
        output_image_path = os.path.join(images_dir, f'augmented_{i}_{i + 1}.jpg')
        output_annotations_path = os.path.join(labels_dir, f'augmented_{i}_{i + 1}.txt')
        
        try:
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)

            annotations1 = load_yolo_annotations(annotations1_path)
            annotations2 = load_yolo_annotations(annotations2_path)

            img1_height, img1_width = image1.shape[:2]
            img2_height, img2_width = image2.shape[:2]

            bboxes1 = [yolo_to_bbox(ann, img1_width, img1_height) for ann in annotations1]
            bboxes2 = [yolo_to_bbox(ann, img2_width, img2_height) for ann in annotations2]

            augmented_image, augmented_bboxes = cut_paste_augmentation(image1, bboxes1, image2, bboxes2, alpha=0.8)
            augmented_annotations = [bbox_to_yolo(bbox, img2_width, img2_height) for bbox in augmented_bboxes]

            cv2.imwrite(output_image_path, augmented_image)
            save_yolo_annotations(output_annotations_path, augmented_annotations)

            augmented_count += 1
            i += 2  # Move to the next pair of images
        
        except FileNotFoundError as e:
            tqdm.write(f"Error: {e}. Skipping this pair of images.")
            i += 2  # Move to the next pair of images
        
        pbar.update(1)

    pbar.close()

# Main function to apply augmentation to the dataset
def main():
    dataset_root_dir = 'dataset'
    num_pairs_to_generate = 3000
    apply_augmentation_to_dataset(dataset_root_dir, num_pairs=num_pairs_to_generate)

if __name__ == "__main__":
    main()
