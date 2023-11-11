# Detecting cracks in sanitary-ware products (ceramics) by select image from disk
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

# Function to perform YOLO detection on an image
def perform_detection(image_path, label):
    frame = cv2.imread(image_path)
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
                label_text = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert frame to RGB (PIL uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to ImageTk format
    image_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

    # Update the label with the new image
    label.config(image=image_tk)
    label.image = image_tk

# Function to handle the "Select Image" button
def select_image(label):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if file_path:
        perform_detection(file_path, label)

# Create the main window
root = tk.Tk()
root.title("YOLOv4 Object Detection")

# Create label for displaying the image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Create button for selecting an image
select_button = tk.Button(root, text="Select Image", command=lambda: select_image(image_label))
select_button.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()
