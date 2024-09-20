import cv2

def check_model_files(cfg_path, weights_path):
    # Load YOLO model
    try:
        net = cv2.dnn.readNet(weights_path, cfg_path)
        print(f"Successfully loaded model files:\n - CFG: {cfg_path}\n - WEIGHTS: {weights_path}")
        
        # Get layer names and output layers
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        print("Layer names:", layer_names)
        print("Output layers:", output_layers)

        return True
    except Exception as e:
        print(f"Error loading model files: {e}")
        return False

# Replace with your actual file paths
cfg_path = 'yolov3_testing.cfg'
weights_path = 'yolov3_training_2000.weights'

if check_model_files(cfg_path, weights_path):
    print("Model files are correctly loaded.")
else:
    print("There was an issue with the model files.")
