from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import logging
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all routes

# Correct paths to your configuration and weights files
cfg_path = r"C:\Users\hussa\Downloads\Weapon_Detection_Project\Real-Time CCTV Video Analysis Deep Learning for Weapon Detection\yolov4.cfg"
weights_path = r"C:\Users\hussa\Downloads\Weapon_Detection_Project\Real-Time CCTV Video Analysis Deep Learning for Weapon Detection\yolov4.weights"

# Debugging: Print paths to ensure they are correct
print(f"CFG Path: {cfg_path}")
print(f"Weights Path: {weights_path}")

# Check if the model files exist
if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")

# Load YOLO model
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define weapon categories (Example categories)
weapon_categories = {
    'gun': ['gun', 'pistol', 'rifle', 'shotgun'],
    'knife': ['knife'],
    'bat': ['bat']
}

def get_weapon_name(label):
    for category, keywords in weapon_categories.items():
        if label.lower() in keywords:
            return category
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_weapon():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Process Image
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(file_path)
                if image is None:
                    os.remove(file_path)
                    return jsonify({'error': 'Could not read image'})
                
                boxes, confidences, class_ids = process_frame(image)
                os.remove(file_path)

                weapon_detected = False
                weapon_name = None
                for i, class_id in enumerate(class_ids):
                    detected_label = classes[class_id]
                    confidence = confidences[i]
                    print(f"Detected {detected_label} with confidence {confidence}")  # Debug info
                    weapon_name = get_weapon_name(detected_label)
                    if weapon_name:
                        weapon_detected = True
                        break

                detection_summary = {
                    'weapon_detected': weapon_detected,
                    'weapon_name': weapon_name if weapon_detected else 'No weapon detected'
                }
                return jsonify({'result': detection_summary, 'frame_count': 1})

            # Process Video
            elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                video_capture = cv2.VideoCapture(file_path)
                if not video_capture.isOpened():
                    os.remove(file_path)
                    return jsonify({'error': 'Could not open video'})

                frame_count = 0
                weapon_detected = False
                weapon_name = None

                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    frame_count += 1
                    boxes, confidences, class_ids = process_frame(frame)

                    for i, class_id in enumerate(class_ids):
                        detected_label = classes[class_id]
                        confidence = confidences[i]
                        print(f"Detected {detected_label} with confidence {confidence}")  # Debug info
                        weapon_name = get_weapon_name(detected_label)
                        if weapon_name:
                            weapon_detected = True
                            break

                    if weapon_detected:
                        break

                video_capture.release()
                os.remove(file_path)

                detection_summary = {
                    'weapon_detected': weapon_detected,
                    'weapon_name': weapon_name if weapon_detected else 'No weapon detected'
                }
                return jsonify({'result': detection_summary, 'frame_count': frame_count})

        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            return jsonify({'error': f'Internal server error: {e}'})

    return jsonify({'error': 'File not processed'})

def process_frame(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            if len(detection) >= 5:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Lower the confidence threshold for testing
                if confidence > 0.5:  # Adjusted for testing
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    return boxes, confidences, class_ids

if __name__ == '__main__':
    app.run(debug=True, port=5000)
