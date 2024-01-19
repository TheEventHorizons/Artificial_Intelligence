# Facial Recognition System

This Python script utilizes the `face_recognition` and `cv2` libraries to perform real-time facial recognition using a webcam. The script captures video frames, recognizes faces, and displays the live feed with recognized faces highlighted.

## Dependencies

Make sure you have the following Python libraries installed:

- `face_recognition`
- `cv2`

You can install them using:

```bash
pip install face_recognition opencv-python
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Update the `known_faces_folder` variable with the path to the folder containing images of known faces.

3. Run the script:

```bash
python your_script_name.py
```

4. The script will open a window displaying the webcam feed with recognized faces highlighted. Press 'q' to quit.

## Note

- Ensure that the images in the `known_faces_folder` are properly formatted (`.png`, `.jpg`, or `.jpeg`).
- The script continuously captures frames from the webcam until you press 'q' to quit.

Feel free to customize the script to fit your specific use case or integrate it into your projects.