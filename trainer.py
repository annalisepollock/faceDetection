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
grayLabels = []
croppedFaces = []
imageLabels = []

cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Walk through each subfolder in the 'images' directory
for idx, subfolder in enumerate(os.listdir(imagesFolder)):
    subfolderPath = os.path.join(imagesFolder, subfolder)
    print("index: ", idx)
    print("subfolder: ", subfolder)
    print()
    # Check if it's a directory
    if os.path.isdir(subfolderPath):
        # Use the subfolder name as the label and assign it a numerical ID
        labelIds[idx] = subfolder

        for file in os.listdir(subfolderPath):
            if file == ".DS_Store":  # Skip .DS_Store
                continue
            filePath = os.path.join(subfolderPath, file)
            print("file: ", file)
            if file.lower().endswith(('.jpg', '.png', '.jfif')):
                pilImage = Image.open(filePath).convert('L')
                imageArray = np.array(pilImage)

                grayscaleImages.append(pilImage)
                grayLabels.append(idx)

                faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    print("here")
                    # Sort faces by size (width * height) in descending order and pick the largest
                    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                    largest_face = faces[0]  # Select the largest face
                    x, y, w, h = largest_face
                    faceRegion = imageArray[y:y + h, x:x + w]  # Crop the face region
                    croppedFaces.append(faceRegion)
                    imageLabels.append(idx)

                count = 0
                for (x, y, w, h) in faces:
                    if count == 0:
                        faceRegion = imageArray[y:y + h, x:x + w]  # Crop the face region
                        croppedFaces.append(faceRegion)
                        imageLabels.append(idx)
                        print("added face", subfolder, "with ID", idx)
                        count += 1

with open("labels.pickle", "wb") as f:
    pickle.dump(labelIds, f)

print("Label-to-ID mapping has been serialized to 'labels.pickle'.")
print("Training the model...")

for face in croppedFaces:
    cv2.imshow("face", face)
    cv2.waitKey(0)
# Step 1: Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Step 2: Train the recognizer with the face images and their corresponding labels
recognizer.train(croppedFaces, np.array(imageLabels))

# Step 3: Save the trained model to a file
recognizer.save("trainer.yml")

print("Model has been trained and saved to 'trainer.yml'.")


