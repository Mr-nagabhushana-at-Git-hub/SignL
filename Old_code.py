import os 
import cv2
import numpy as np
import mediapipe as mp
from fer import FERell
from matplotlib import pyplot as plt
import face_recognition  # Import face recognition library

# Set environment variable to disable oneDNN custom operations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

# Initialize the emotion detector
emotion_detector = FER(mtcnn=True)

# Load known faces and their names
KNOWN_FACES_DIR = r"D:\AI_Team\majorSignL\src\data\fase_data"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(filename)[0])  # Use file name as name

# Function to process images with MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image.flags.writeable = False  # Prevent modifications
    results = model.process(image)  # Process the image
    image.flags.writeable = True  # Allow modifications
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results

# Function to draw styled landmarks, detect emotions, and perform face recognition
def process_frame(image, face_results, holistic_results):
    # Define styles for different landmarks
    face_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1)
    hand_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    # Resize frame for face recognition (to improve performance)
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert to RGB
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Scale face locations back to the original frame size
        top, right, bottom, left = [v * 4 for v in face_location]
        
        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            image, name, (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
    
    # Draw face landmarks
    if face_results and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=face_spec
            )
    
    # Draw left hand landmarks
    if holistic_results and holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
            hand_spec
        )
    
    # Draw right hand landmarks
    if holistic_results and holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
            hand_spec
        )

# Initialize video capture
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face and holistic landmarks
        face_results = face_mesh.process(rgb_frame)
        _, holistic_results = mediapipe_detection(rgb_frame, holistic)
        
        # Process frame for landmarks, emotion detection, and face recognition
        process_frame(frame, face_results, holistic_results)
        
        # Display the output frame
        cv2.imshow('Face Recognition & Emotion Detection', frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()