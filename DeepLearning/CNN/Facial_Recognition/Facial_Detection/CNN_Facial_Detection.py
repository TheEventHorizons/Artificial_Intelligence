import face_recognition
import cv2
import os
import re

def load_known_faces(folder_path):
    """
    Load known faces and their encodings from a specified folder.

    Parameters:
    - folder_path: The path to the folder containing images of known faces.

    Returns:
    - known_face_encodings: List of face encodings for known faces.
    - known_face_names: List of corresponding file names (person names) for known faces.
    """
    known_face_encodings = []
    known_face_names = []

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        # Create the full file path
        img_path = os.path.join(folder_path, filename)

        # Check if the file is a valid image
        if os.path.isfile(img_path) and re.match(r'^\.(png|jpe?g|gif)$', os.path.splitext(filename)[1]):
            # Load the image and obtain the face encoding
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)

            # Ensure at least one face is detected
            if len(encoding) > 0:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.basename(img_path))

    return known_face_encodings, known_face_names

def recognize_faces(frame, known_face_encodings, known_face_names):
    """
    Recognize faces in a video frame using known face encodings.

    Parameters:
    - frame: The video frame to process.
    - known_face_encodings: List of face encodings for known faces.
    - known_face_names: List of corresponding file names (person names) for known faces.

    Returns:
    - face_locations: List of face locations in the frame.
    - face_names: List of corresponding recognized person names.
    """
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(os.path.splitext(name)[0])

    return face_locations, face_names

def draw_faces(frame, face_locations, face_names):
    """
    Draw rectangles and labels around recognized faces on the video frame.

    Parameters:
    - frame: The video frame to draw on.
    - face_locations: List of face locations in the frame.
    - face_names: List of corresponding recognized person names.
    """
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (3, 168, 124), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (3, 168, 124), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

if __name__ == '__main__':
    # Folder containing images of known faces
    known_faces_folder = "/path/folder/"

    # Load known faces
    known_face_encodings, known_face_names = load_known_faces(known_faces_folder)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Recognize faces in the frame
        face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)

        # Draw recognized faces on the frame
        draw_faces(frame, face_locations, face_names)

        # Display the resulting image
        cv2.imshow('user-cam', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
