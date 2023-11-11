# Detecting cracks in sanitary-ware products (ceramics) in real-time
import cv2
import numpy as np

cfg_path = "yolov4/yolov4-tiny-custom.cfg"
weights_path = "yolov4/yolov4-tiny-custom_best.weights"
# Define the net object
net = cv2.dnn.readNet(cfg_path, weights_path)

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open("yolov4/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# Open a video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Convert frame to blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO predictions
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                color = (0, 255, 0)  # Green by default
                if classes[class_id] == "Crack":
                    color = (0, 0, 255)  # Red for cracks
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('YOLOv4 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()