import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pygame
import os

emotion = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the trained model
model = keras.models.load_model("model_trained.h5")

# Initialize pygame mixer
pygame.mixer.init()

# Define the paths to your music files based on emotions
music_folder = "music"
music_files = {
    'Anger': 'anger_music.mp3',
    'Disgust': 'disgust_music.mp3',
    'Fear': 'fear_music.mp3',
    'Happy': 'happy_music.mp3',
    'Sad': 'sad_music.mp3',
    'Surprise': 'surprise_music.mp3',
    'Neutral': 'neutral_music.mp3'
}

# Load music files
emotion_music = {}
for emotion, file in music_files.items():
    emotion_music[emotion] = os.path.join(music_folder, file)

# Load the cascade classifier for detecting faces
face_cas = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# Initialize the video capture
cam = cv2.VideoCapture(0)

# Define the font for displaying text on the frame
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cam.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cas.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract the face component and resize it to match the input size of the model
            face_component = gray[y:y+h, x:x+w]
            fc = cv2.resize(face_component, (48, 48))
            inp = np.reshape(fc, (1, 48, 48, 1)).astype(np.float32)
            inp = inp / 255.

            # Make prediction using the loaded model
            prediction = model.predict(inp)
            emotion_detected = emotion[np.argmax(prediction)]

            # Play music based on the detected emotion
            music_file = emotion_music.get(emotion_detected)
            if music_file:
                pygame.mixer.music.load(music_file)
                pygame.mixer.music.play(-1)  # Play the music in a loop

            # Display the emotion label and confidence percentage on the frame
            cv2.putText(frame, emotion_detected, (x, y), font, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("image", frame)

        # Break the loop if ESC is pressed
        if cv2.waitKey(1) == 27:
            break
    else:
        print('Error')

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
