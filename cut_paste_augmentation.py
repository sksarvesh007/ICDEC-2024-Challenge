# Trying the cut paste augmentation to increase the dataset size
#did some poison blending to adjust the cut pasted images to the background and the image looks more realistic now
import cv2
import numpy as np
import os

#Loading the image annotations from the file
def load_yolo_annotations(file_path):
    with open(file_path, 'r') as f:
        annotations = [line.strip().split() for line in f.readlines()]
    return annotations

#Saving the image annotations to the file
def save_yolo_annotations(file_path, annotations):
    with open(file_path, 'w') as f:
        for annotation in annotations:
            line = ' '.join([str(x) for x in annotation])
            f.write(line + '\n')
            
#the yolo annotations are in the format of class x_center y_center width height
def yolo_to_bbox(yolo_annotation, img_width, img_height):
    class_id , x_centre , y_centre , width , height = map(float, yolo_annotation)
    x_centre *= img_width
    y_centre *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_centre - width/2)
    x2 = int(x_centre + width/2)
    y1 = int(y_centre - height/2)
    y2 = int(y_centre + height/2)
    return int(class_id), x1, y1, x2, y2

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

#function to make sure that the randomly selected bounding box is not overlapping the other bounding box in the second image 
def check_overlap(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    if x1_min >= x2_max or x1_max <= x2_min or y1_min >= y2_max or y1_max <= y2_min:
        return False
    return True

# finding the matching class height to adjust the bounding box in the second image
def find_matching_class_height(bboxes, class_id):
    for bbox in bboxes:
        if bbox[0] == class_id:
            return (bbox[2] + bbox[3]) // 2
    return None

# function for the cut paste augmentation 
def cut_paste_augmentation(image1, bboxes1, image2, bboxes2 , alpha = 0.5):
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
        
        # Try placing the patch at a random x position until it does not overlap
        max_attempts = 100
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
            print(f"Failed to place patch for bbox {bbox} without overlap after {max_attempts} attempts.")
    
    return img2, bboxes2
#main function 
def main(image1_path, annotations1_path, image2_path, annotations2_path, output_image_path, output_annotations_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    annotations1 = load_yolo_annotations(annotations1_path)
    annotations2 = load_yolo_annotations(annotations2_path)
    
    img1_height, img1_width = image1.shape[:2]
    img2_height, img2_width = image2.shape[:2]
    
    bboxes1 = [yolo_to_bbox(ann, img1_width, img1_height) for ann in annotations1]
    bboxes2 = [yolo_to_bbox(ann, img2_width, img2_height) for ann in annotations2]\
    
    augmented_image, augmented_bboxes = cut_paste_augmentation(image1, bboxes1, image2, bboxes2)
    augmented_annotations = [bbox_to_yolo(bbox, img2_width, img2_height) for bbox in augmented_bboxes]
    
    cv2.imwrite(output_image_path, augmented_image)
    save_yolo_annotations(output_annotations_path, augmented_annotations)
    
    print(f"Augmented image saved to {output_image_path}")
    
image1_path = 'dataset/images/train/night (1).jpg'
annotations1_path = 'dataset/labels/train/night (1).txt'
image2_path = 'dataset/images/train/night (2).jpg'
annotations2_path = 'dataset/labels/train/night (1).txt'
output_image_path = 'cut_paste_output_image.jpg'
output_annotations_path = 'cut_paste_output_image.txt'

main(image1_path, annotations1_path, image2_path, annotations2_path, output_image_path, output_annotations_path)