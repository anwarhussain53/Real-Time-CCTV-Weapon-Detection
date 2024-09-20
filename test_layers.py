import cv2

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")

# Print all layer names
layer_names = net.getLayerNames()
print("Layer names:", tuple(layer_names))

# Print unconnected output layers
unconnected_out_layers = net.getUnconnectedOutLayers()
print("Unconnected output layers:", unconnected_out_layers)

# Print output layer names
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
print("Output layers:", output_layers)
