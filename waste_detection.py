#!pip3 install torch torchvision
#!pip install torch torchvision opencv-python
# Run the above commands to install the iibraries
import torch

import cv2
import numpy as np

# Load the trained model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")

# Set the source for the camera feed. This value depends on the number of cameras connected ot the system
source = 0

# Open the camera
cap = cv2.VideoCapture(source)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # predicting objects from the trained model. it returns a list of detected objetcs with bounding box coodinates
    results = model(frame)
    detected_objects = results.pred[0]
    detections = detected_objects[detected_objects[:, 4] > 0.5]

    # Draw rectangles around the detected objects
    for detection in detections:
        x_center, y_center, width, height, conf, class_idx = detection
        class_idx = int(class_idx)
        class_name = model.names[class_idx]

        cv2.rectangle(frame, (int(x_center), int(y_center)), (int(width), int(height)), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (int(x_center), int(y_center) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the detected faces
    cv2.imshow('Waste Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
