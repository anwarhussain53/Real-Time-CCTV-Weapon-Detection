from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)

# Ensure the temp directory exists
if not os.path.exists('./temp'):
    os.makedirs('./temp')

# Load the YOLO model
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Gun", "Knife"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            print("No file provided")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        file_path = f"./temp/{file.filename}"
        file.save(file_path)
        print(f"File saved at {file_path}")

        # Process video file
        video_capture = cv2.VideoCapture(file_path)
        if not video_capture.isOpened():
            print("Unable to open video")
            raise Exception('Unable to open video')

        results = []
        frame_count = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                print("No more frames or error reading frame")
                break

            frame_count += 1
            height, width, channels = frame.shape

            # Detect objects in the frame
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            detected_classes = [classes[class_ids[i]] for i in indexes.flatten()] if len(indexes) > 0 else []

            results.append(detected_classes)
        
        video_capture.release()
        return jsonify({'result': results, 'frame_count': frame_count})

    except Exception as e:
        print(f'Exception: {str(e)}')
        return jsonify({'error': f'Exception: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)
