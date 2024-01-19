# Import necessary libraries
import os
import re
import cv2
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras

# Function to load known faces from a folder
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        # Check if the file is an image (png, jpg, jpeg, or gif extension)
        if os.path.isfile(img_path) and re.match(r'^\.(png|jpe?g|gif)$', os.path.splitext(filename)[1]):
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)

            # Check if the face was successfully encoded
            if len(encoding) > 0:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

# Function to predict emotion from a face image
def predict_emotion(face_image):
    
    # Resize the image to the size expected by the model (e.g., 48x48)
    resized_image = cv2.resize(face_image, (48, 48))

    # Add a dimension to match the expected shape by the model
    input_image = np.expand_dims(resized_image, axis=0)

    # Make prediction with the model
    predictions = loaded_model.predict(input_image)

    # Get the predicted class and probabilities
    predicted_class = np.argmax(predictions)

    return predicted_class

# Function to recognize faces in an image
def recognize_faces(frame, known_face_encodings, known_face_names):
    # Resize the image to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Locate face locations in the resized image
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare detected faces with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            # Find the first matching known face
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(os.path.splitext(name)[0])

    return face_locations, face_names

# Function to draw recognized faces on an image
def draw_faces(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Extract the face image for emotion prediction
        face_image = frame[top:bottom, left:right]
        predicted_class = predict_emotion(face_image)
        name_with_emotion = f'{name} ({class_names[predicted_class]})'

        # Draw a rectangle around the face and display the name with emotion
        cv2.rectangle(frame, (left, top), (right, bottom), (3, 168, 124), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (3, 168, 124), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name_with_emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Set the path for training data
data_dir_train = '/your/folder/Who_is_sad/archive/train'

batch_size = 40

# Set the size of the images
img_size = (48, 48)

# Load training data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
)

class_names = train_data.class_names
print(class_names)

# If executed as a script
if __name__ == '__main__':
    known_faces_folder = "/your/folder/Who_is_sad/known_faces/"
    known_face_encodings, known_face_names = load_known_faces(known_faces_folder)

    model_path = '/your/folder/Who_is_sad/archive/reduced/models/best-model.h5'
    loaded_model = load_model(model_path)

    # Video capture from webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture an image from the webcam
        ret, frame = video_capture.read()

        # Recognize faces and predict emotions
        face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)
        draw_faces(frame, face_locations, face_names)

        # Display the image with detected faces and emotions
        cv2.imshow('user-cam', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    video_capture.release()
    cv2.destroyAllWindows()
