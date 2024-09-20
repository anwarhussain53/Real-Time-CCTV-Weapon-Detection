import cv2

# Path to the image file
image_path = 'C:/Users/hussa/Downloads/Weapon_Detection_Project/Real-Time CCTV Video Analysis Deep Learning for Weapon Detection/uploads/test_images/test_image.jpg'

# Read the image file
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image.")
else:
    print("Image loaded successfully.")
