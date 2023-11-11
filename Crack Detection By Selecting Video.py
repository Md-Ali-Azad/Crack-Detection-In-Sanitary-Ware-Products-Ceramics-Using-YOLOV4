# Detecting cracks in sanitary-ware products (ceramics) by select video from disk
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

cfg_path = "yolov4/yolov4-tiny-custom.cfg"
weights_path = "yolov4/yolov4-tiny-custom_best.weights"

# Define the net object
net = cv2.dnn.readNet(cfg_path, weights_path)

# Load class names
with open("yolov4/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform YOLO detection on a video
def perform_detection(video_path, label):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Set up VideoWriter for saving the output video with default codec
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height), isColor=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()  # Keep a copy of the original frame

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
                    label_text = f"{classes[class_id]}: {confidence:.2f}"
                    print(label_text)
                    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the processed frame
        cv2.imshow('YOLOv4 Object Detection', frame)

        # Write the frame to the output video file
        out.write(frame)

        # Check for user input to exit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release the VideoWriter and capture objects
    out.release()
    cap.release()

# Function to handle the "Select Video" button
def select_video(label):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        perform_detection(file_path, label)

# Create the main window
root = tk.Tk()
root.title("YOLOv4 Object Detection on Video")

# Create label for displaying the video frames
video_label = tk.Label(root)
video_label.pack(pady=10)

# Create button for selecting a video
select_button = tk.Button(root, text="Select Video", command=lambda: select_video(video_label))
select_button.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()
