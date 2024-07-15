from ultralytics import YOLO

# Load your pre-trained YOLOv8 model
model_path = 'yolo 8x model/train/weights/best.pt'
model = YOLO(model_path)

# Perform validation
results = model.val(data='C:/Users/SARVESH/Desktop/repos/ICDEC_2024_Challenge/dataset.yaml', imgsz=640, conf=0.001)

# Retrieve and print the mAP scores
maps = results.maps()
map50 = maps['0.5']  # mAP at IoU=0.5
map5095 = maps['0.5:0.95']  # mAP at IoU=0.5:0.95

print(f'mAP@0.5 = {map50:.4f}')
print(f'mAP@0.5:0.95 = {map5095:.4f}')

# Retrieve and print the AP for each class
for class_id, ap in results.class_map().items():
    print(f'Class {class_id}: AP = {ap:.4f}')
