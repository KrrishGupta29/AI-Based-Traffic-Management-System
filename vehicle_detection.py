'''
import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg.txt")

# Load COCO class names
with open("coco.names.txt", "r") as f:
    classes = f.read().strip().split("\n")

# Load input image
image_path = "car.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape

# Preprocess image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
detections = net.forward(output_layers)

# Count detected vehicles (class_id == 2 for cars)
vehicle_count = 0

for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and class_id == 2:
            vehicle_count += 1  # Increment count for each car

            # Draw bounding box (optional)
            center_x = int(obj[0] * width)
            center_y = int(obj[1] * height)
            w = int(obj[2] * width)
            h = int(obj[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Write vehicle count to file
with open("vehicle_count.txt", "w") as f:
    f.write(str(vehicle_count))

print(f"Detected Vehicles: {vehicle_count} (written to vehicle_count.txt)")

# Show result
cv2.imshow("Vehicle Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
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

