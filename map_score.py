from ultralytics import YOLO

# Load your pre-trained YOLOv8 model
model_path = 'yolo 8x model/train/weights/best.pt'
model = YOLO(model_path)

# Perform validation
results = model.val(data='dataset', imgsz=640, conf=0.001)

# Print the mAP for each class
for class_id, ap in results['class_map'].items():
    print(f'Class {class_id}: AP = {ap:.4f}')

# Print the overall mAP
print(f'mAP@0.5 = {results["map50"]:.4f}')
print(f'mAP@0.5:0.95 = {results["map"]:.4f}')