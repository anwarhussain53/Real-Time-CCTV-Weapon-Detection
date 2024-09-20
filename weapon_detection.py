import cv2
import numpy as np

# Load YOLO network
net = cv2.dnn.readNet(
    r"C:\Users\hussa\Downloads\Weapon_Detection_Project\Real-Time CCTV Video Analysis Deep Learning for Weapon Detection\yolov4.cfg",
    r"C:\Users\hussa\Downloads\Weapon_Detection_Project\Real-Time CCTV Video Analysis Deep Learning for Weapon Detection\yolov4.weights"
)

# Load class names
with open(r"C:\Users\hussa\Downloads\Weapon_Detection_Project\Real-Time CCTV Video Analysis Deep Learning for Weapon Detection\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

# Fix for net.getUnconnectedOutLayers() returning scalar values
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load image
image = cv2.imread(r"C:\Users\hussa\Downloads\Weapon_Detection_Project\Real-Time CCTV Video Analysis Deep Learning for Weapon Detection\uploads\test_images\test_image.jpg")

# Check if the image was loaded
if image is None:
    raise ValueError("Could not load image. Please check the path.")

height, width, _ = image.shape

# Preprocess the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Process each detection
boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Adjust this threshold if needed
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the rectangle box coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression to remove duplicate boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw the bounding boxes
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the result image
output_path = r"C:\Users\hussa\Downloads\Weapon_Detection_Project\Real-Time CCTV Video Analysis Deep Learning for Weapon Detection\uploads\test_images\result.jpg"
cv2.imwrite(output_path, image)

print(f"Results saved to {output_path}")
