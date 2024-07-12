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

# Cut-paste augmentation function with poison blending
def cut_paste_augmentation(image1, bbox, image2, bboxes2, alpha=0.8):
    img1 = image1.copy()
    img2 = image2.copy()
    
    class_id, x1, y1, x2, y2 = bbox
    patch = img1[y1:y2, x1:x2]

    # Try placing the patch at a random x position until it does not overlap
    max_attempts = 1000
    placed = False
    for _ in range(max_attempts):
        new_x1 = np.random.randint(0, img2.shape[1] - (x2 - x1))
        new_y1 = np.random.randint(0, img2.shape[0] - (y2 - y1))
        new_x2 = new_x1 + (x2 - x1)
        new_y2 = new_y1 + (y2 - y1)

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
    
    if len(image_files) < 5:
        print("Error: There are not enough images in the directory to perform augmentation.")
        return
    
    augmented_count = 0
    i = 0
    pbar = tqdm(total=num_pairs)
    while augmented_count < num_pairs and i < len(image_files) - 5:
        image1_path = os.path.join(images_dir, image_files[i])
        image_paths = [os.path.join(images_dir, image_files[i + j + 1]) for j in range(4)]
        
        annotations1_path = os.path.join(labels_dir, image_files[i].replace('.jpg', '.txt').replace('.jpeg', '.txt'))
        annotation_paths = [os.path.join(labels_dir, image_files[i + j + 1].replace('.jpg', '.txt').replace('.jpeg', '.txt')) for j in range(4)]
        
        output_image_paths = [os.path.join(images_dir, f'augmented_{i}_{i + j + 1}.jpg') for j in range(4)]
        output_annotation_paths = [os.path.join(labels_dir, f'augmented_{i}_{i + j + 1}.txt') for j in range(4)]
        
        try:
            image1 = cv2.imread(image1_path)
            images = [cv2.imread(image_path) for image_path in image_paths]

            annotations1 = load_yolo_annotations(annotations1_path)
            annotations = [load_yolo_annotations(annotation_path) for annotation_path in annotation_paths]

            img1_height, img1_width = image1.shape[:2]
            img_heights_widths = [(image.shape[:2]) for image in images]

            bboxes1_class0 = [yolo_to_bbox(ann, img1_width, img1_height) for ann in annotations1 if int(ann[0]) == 0]
            bboxes1_non_class0 = [yolo_to_bbox(ann, img1_width, img1_height) for ann in annotations1 if int(ann[0]) != 0]
            bboxes_list = [[yolo_to_bbox(ann, img_width, img_height) for ann in anns] for anns, (img_height, img_width) in zip(annotations, img_heights_widths)]

            for j in range(4):
                img, bboxes = images[j], bboxes_list[j]
                bboxes_class0 = [bbox for bbox in bboxes if bbox[0] == 0]
                bboxes_non_class0 = [bbox for bbox in bboxes if bbox[0] != 0]

                for bbox in bboxes1_non_class0:
                    img, bboxes = cut_paste_augmentation(image1, bbox, img, bboxes_class0 + bboxes_non_class0, alpha=0.8)
                
                augmented_annotations = [bbox_to_yolo(bbox, img_heights_widths[j][1], img_heights_widths[j][0]) for bbox in bboxes]

                cv2.imwrite(output_image_paths[j], img)
                save_yolo_annotations(output_annotation_paths[j], augmented_annotations)

            augmented_count += 4
            i += 5  # Move to the next set of images
        
        except FileNotFoundError as e:
            tqdm.write(f"Error: {e}. Skipping this set of images.")
            i += 5  # Move to the next set of images
        
        pbar.update(4)

    pbar.close()

# Main function to apply augmentation to the dataset
def main():
    dataset_root_dir = 'dataset'
    num_pairs_to_generate = 3000
    apply_augmentation_to_dataset(dataset_root_dir, num_pairs=num_pairs_to_generate)

if __name__ == "__main__":
    main()
