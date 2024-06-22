import cv2
import os
from ultralytics import YOLO

def get_dark_colors(num_classes):
    """Generate a list of dark colors for each class"""
    dark_colors = [
        (139, 0, 0), (0, 100, 0), (0, 0, 139), (85, 107, 47),
        (139, 69, 19), (0, 139, 139), (139, 0, 139), (47, 79, 79),
        (112, 128, 144), (75, 0, 130), (123, 104, 238), (0, 128, 128),
        (128, 0, 0), (34, 139, 34), (70, 130, 180), (105, 105, 105)
    ]
    if num_classes > len(dark_colors):
        dark_colors *= (num_classes // len(dark_colors)) + 1
    return dark_colors[:num_classes]

def main(video_path, model_path, output_path=None):
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Get dark colors for each class
    num_classes = len(model.names)
    colors = get_dark_colors(num_classes)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video writer initialized to save the output video (if output_path is provided)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results:
            for detection in result.boxes.data:
                x1, y1, x2, y2, conf, cls = detection
                cls = int(cls)
                label = f"{model.names[cls]}"
                color = colors[cls]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the frame
        cv2.imshow('YOLO Detection', frame)

        # Write the frame to the output video (if output_path is provided)
        if output_path:
            out.write(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"  # Replace with the path to your video file
    model_path = "last.pt"    # Replace with the path to your YOLO model checkpoint
    output_path = "output_video.mp4"  # Replace with the path to save the output video (optional)

    main(video_path, model_path, output_path)
