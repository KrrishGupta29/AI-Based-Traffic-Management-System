
from ultralytics import YOLO
import cv2
import sys

# Set this if called from Flask
run_from_flask = len(sys.argv) > 1 and sys.argv[1] == "--from-flask"

# Load YOLOv8 model (Nano is fast, you can also use 'yolov8s.pt')
model = YOLO("yolov8s.pt")

# Load the image (update this to the correct path if from Flask)
image_path = "static/uploaded.png" if run_from_flask else "Screenshot.png"
results = model(image_path)

# Count detected vehicles
vehicle_classes = ["car", "motorbike", "bus", "truck"]
vehicle_count = 0

# Annotated image
image = cv2.imread(image_path)

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name in vehicle_classes:
            vehicle_count += 1
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the output image
cv2.imwrite("static/output.png", image)

# Save vehicle count for use in Flask
with open("vehicle_count.txt", "w") as f:
    f.write(str(vehicle_count))

# Show the image only when NOT running from Flask
if run_from_flask:
    # Save image and skip showing window
    cv2.imwrite("static/output.png", image)
    import subprocess
    subprocess.Popen(['python', 'show_image.py'])
else:
    # Show image for manual runs
    cv2.imshow("Vehicle Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()