import cv2

video_path = r"C:\Users\hussa\Downloads\Weapon_Detection_Project\Real-Time CCTV Video Analysis Deep Learning for Weapon Detection\ak47.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file or webcam. Path: {video_path}")
else:
    print(f"Video file opened successfully. Path: {video_path}")

cap.release()
