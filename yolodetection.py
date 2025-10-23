from ultralytics import YOLO
import numpy 

model = YOLO("yolov8n.pt", "v8")

# Load the image
detection_output = model.predict(r"C:\Users\CHAITANYA\OneDrive\Desktop\dog1.jpeg")

print(detection_output)