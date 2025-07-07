'''
import cv2
import numpy as np
import sys
run_from_flask = '--from-flask' in sys.argv
# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg.txt")

# Load class names
with open("coco.names.txt", "r") as f:
    classes = f.read().strip().split("\n")

# Load image
image_path = "static/uploaded.png"
image = cv2.imread(image_path)

height, width, _ = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Data to hold detections
boxes = []
confidences = []
class_ids = []

# Process YOLO outputs
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter for vehicles only (e.g. class_id == 2 for car)
        if confidence > 0.5 and class_id == 2:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Count filtered vehicles and draw boxes
vehicle_count = len(indexes)

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save vehicle count to file
with open("vehicle_count.txt", "w") as f:
    f.write(str(vehicle_count))

print(f"Detected Vehicles (after NMS): {vehicle_count}")

# Show image
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

'''
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