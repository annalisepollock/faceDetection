import os
from PIL import Image
import cv2
import numpy as np
import pickle

# Get the directory of the current file
currentDir = os.path.dirname(os.path.abspath(__file__))

# Append the 'images' folder to the current directory
imagesFolder = os.path.join(currentDir, 'images')

# Initialize a dictionary to store label-to-ID mappings
labelIds = {}

grayscaleImages = []
croppedFaces = []
imageLabels = []

cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Walk through each subfolder in the 'images' directory
for idx, subfolder in enumerate(os.listdir(imagesFolder)):
    subfolderPath = os.path.join(imagesFolder, subfolder)
    
    # Check if it's a directory
    if os.path.isdir(subfolderPath):
        # Use the subfolder name as the label and assign it a numerical ID
        labelIds[subfolder] = idx

        for file in os.listdir(subfolderPath):
            filePath = os.path.join(subfolderPath, file)
            
            if file.lower().endswith(('.jpg', '.png', '.jfif')):
                pilImage = Image.open(filePath).convert('L')
                imageArray = cv2.cvtColor(np.array(pilImage), cv2.COLOR_GRAY2BGR)

                grayscaleImages.append(pilImage)
                imageLabels.append(idx)

                faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    faceRegion = pilImage.crop((x, y, x + w, y + h))  # Crop the face region
                    croppedFaces.append(faceRegion)
                    imageLabels.append(idx)

with open("labels.pickle", "wb") as f:
    pickle.dump(labelIds, f)

print("Label-to-ID mapping has been serialized to 'labels.pickle'.")

# Step 1: Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Step 2: Train the recognizer with the face images and their corresponding labels
recognizer.train(faceRegion, np.array(imageLabels))

# Step 3: Save the trained model to a file
recognizer.save("trainer.yml")

print("Model has been trained and saved to 'trainer.yml'.")


