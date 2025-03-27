import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import pickle
import pyrealsense2 as rs
import time

# Function to draw eyes on a separate window
def drawEyes(canvas):
    # Draw the eyes (same logic you had before)
    canvas.delete("all")
    canvas.create_oval(100, 100, 200, 300, fill='white', outline='black')
    canvas.create_oval(300, 100, 400, 300, fill='white', outline='black')
    canvas.create_oval(150, 175, 200, 225, fill='black')
    canvas.create_oval(350, 175, 400, 225, fill='black')

# Function to update the name in the face recognition window
def updateName(canvas, name):
    canvas.delete("all")
    canvas.create_text(250, 250, text=name, font=("Arial", 30))

# Step 1: Load the pre-trained Haar face detector and face recognition model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the LBPH Face Recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')  # Replace with the path to your trained model

# Load the labels pickle file
with open('labels.pickle', 'rb') as file:
    data = pickle.load(file)
data = {v: k for k, v in data.items()}

# Set up the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Set up the Tkinter window for face recognition
root = tk.Tk()
root.title("Stranger Danger")
canvas = tk.Canvas(root, width=500, height=500)
canvas.pack()
drawEyes(canvas)

last_recognized_label = None
last_update_time = 0
recognition_delay = 3  # 3 seconds delay for updating

# This function runs the OpenCV code in a separate thread to avoid blocking the GUI
def capture_faces():
    global last_recognized_label, last_update_time, current_time

    try:
        while True:
            current_time = time.time()
            # Wait for a coherent frame
            frames = pipeline.wait_for_frames()

            # Get the color frame
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert to numpy array for OpenCV processing
            color_image = np.asanyarray(color_frame.get_data())

            # Convert the frame to grayscale
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if(len(faces) == 0 and current_time - last_update_time >= recognition_delay):
                drawEyes(canvas)
            # Loop over the detected faces
            for (x, y, w, h) in faces:
                # Extract the region of interest (ROI) for face recognition
                roi_gray = gray[y:y + h, x:x + w]
                # Use LBPH recognizer to predict the label and confidence
                label, confidence = recognizer.predict(roi_gray)

                # If the label has changed and 3 seconds have passed, update the name
                if confidence < 80 and (last_recognized_label != label or current_time - last_update_time >= recognition_delay):
                    # Update the name on the canvas
                    name = data.get(label, "Unknown")
                    updateName(canvas, name)
                    print(f"Recognized: {name} with confidence: {confidence}")

                    # Update the last recognized label and the last update time
                    last_recognized_label = label
                    last_update_time = current_time
                elif confidence >= 80:
                    # If the confidence is too low, show "Stranger Danger"
                    updateName(canvas, "Stranger Danger")
                    print("Not confident")

                # Draw bounding box around the face
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the resulting frame using OpenCV's imshow function
            cv2.imshow("Face Recognition", color_image)

            # If 'q' is pressed, close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline and close OpenCV window
        pipeline.stop()
        cv2.destroyAllWindows()

# Start the face capturing thread
face_thread = threading.Thread(target=capture_faces, daemon=True)
face_thread.start()

# Run the Tkinter main loop
root.mainloop()