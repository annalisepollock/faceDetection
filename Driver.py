import tkinter as tk
from tkinter import Canvas
import cv2
import pickle
import queue
import numpy as np
import time
import threading

# Step 1: Load the Haar Cascade face detector
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Step 2: Load the trained LBPH face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")  # Load the trained model from the .yml file

# Step 3: Load the label mappings from the pickle file
with open("labels.pickle", "rb") as f:
    labelIds = pickle.load(f)

# Reverse the label-to-ID mapping for easier lookup (ID -> label)
idToLabel = {v: k for k, v in labelIds.items()}

window = tk.Tk()
window.title("Face Recognition")
window.geometry("800x600")

# Step 2: Create a Canvas to display recognition results
global canvas
canvas = Canvas(window, width=800, height=400, bg="white")
canvas.pack()

# Step 3: Initialize a queue to handle text updates
textQueue = queue.Queue()
textQueue.put("TEST")
frameQueue = queue.Queue()

# Step 4: Function to draw animated eyes when no face is detected
def drawEyes():
    canvas.delete("all")  # Clear the canvas
    # Draw two circles to represent eyes
    canvas.create_oval(300, 150, 350, 200, fill="black")  # Left eye
    canvas.create_oval(450, 150, 500, 200, fill="black")  # Right eye
    canvas.create_text(400, 300, text="No face detected", font=("Arial", 24), fill="red")

# Step 5: Function to update the displayed text based on recognized faces
def updateText(canvas):
    try:
        # Get the latest text from the queue
        print("textQueue: ", textQueue)
        text = textQueue.get_nowait()
        print("text: ", text)
        canvas.delete("all")  # Clear the canvas
        canvas.create_text(400, 200, text=text, font=("Arial", 46), fill="green")
    except queue.Empty:
        # If the queue is empty, draw animated eyes
        drawEyes()
    # Schedule the function to run again after 100ms
    #window.after(1000, updateText)

def displayFrames():
    if not frameQueue.empty():
        frame = frameQueue.get()
        cv2.imshow("Video Stream - Face Detection", frame)

    # Schedule the function to run again after 10ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        window.quit()  # Close the GUI
    else:
        window.after(10, displayFrames)

def videoProcessing():
    videoCapture = cv2.VideoCapture(0)  # Open the default camera
    lastRecognizedName = None
    lastRecognizedTime = time.time()

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Convert the frame to grayscale
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        recognizedName = None

        if faces is not None:  # Ensure faces is not empty
            for (x, y, w, h) in faces:
                # Draw bounding box around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Predict the face
                faceRegion = grayFrame[y:y + h, x:x + w]
                labelId, confidence = recognizer.predict(faceRegion)

                if confidence < 50:  # Confidence threshold
                    recognizedName = idToLabel[labelId]
                else:
                    recognizedName = "Stranger Danger"
        else:
            recognizedName = None
        # Handle name changes with a delay
        currentTime = time.time()
        if recognizedName != lastRecognizedName:
            if currentTime - lastRecognizedTime >= 3:  # 3-second delay
                lastRecognizedName = recognizedName
                lastRecognizedTime = currentTime
                if recognizedName:
                    textQueue.put(recognizedName)

        elif faces is None:
            if currentTime - lastRecognizedTime >= 3:  # No face detected for 3 seconds
                lastRecognizedName = None
                textQueue.put("")

        if not frameQueue.full():
            frameQueue.put(frame)

    # Release resources
    videoCapture.release()


def main():
   # Start the video processing thread
    #videoThread = threading.Thread(target=videoProcessing, daemon=True)
    #videoThread.start()
    # Start displaying frames
    #displayFrames()
    canvas = Canvas(window, width=800, height=400, bg="white")
    canvas.pack()
    
    updateText(canvas)

    window.mainloop()

    # Destroy OpenCV windows when the GUI is closed
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()