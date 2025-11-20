# src/majorSignL/models/face_processor.py
import cv2
import numpy as np
from pathlib import Path
import face_recognition
import os
import pickle
import logging
from signl.config import FACE_DATA_DIR, CACHE_DIR, FACE_ENCODINGS_CACHE

logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self, data_path: Path):
        """Initialize face recognition with known faces from fase_data folder structure"""
        self.fase_data_path = FACE_DATA_DIR
        self.cache_path = CACHE_DIR
        
        # Create cache directory if it doesn't exist
        self.cache_path.mkdir(exist_ok=True)
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load known faces from fase_data person folders
        self._load_known_faces_from_person_folders()
        
        logger.info(f"[FaceProcessor] Loaded {len(self.known_face_names)} known faces")
    
    def _load_known_faces_from_person_folders(self):
        """Load face encodings from person-named folders in fase_data"""
        encodings_cache = FACE_ENCODINGS_CACHE
        
        # Try to load cached encodings first
        if encodings_cache.exists():
            try:
                with open(encodings_cache, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.known_face_encodings = cached_data['encodings']
                    self.known_face_names = cached_data['names']
                    logger.info(f"[FaceProcessor] Loaded cached encodings for {len(self.known_face_names)} faces")
                    return
            except Exception as e:
                logger.warning(f"[FaceProcessor] Cache load failed: {e}")
        
        # Load faces from person folders
        if not self.fase_data_path.exists():
            logger.error(f"[FaceProcessor] fase_data path not found: {self.fase_data_path}")
            return
        
        # Iterate through person folders
        for person_folder in self.fase_data_path.iterdir():
            if person_folder.is_dir() and not person_folder.name.endswith('.pkl'):
                person_name = person_folder.name
                logger.info(f"[FaceProcessor] Processing person: {person_name}")
                
                # Get all image files from this person's folder
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(person_folder.glob(ext))
                
                person_encodings = []
                successful_loads = 0
                
                # Process first 5 images for each person (for performance)
                #for img_file in image_files[:5]:
                for img_file in image_files:#i want all photos to be loaded
                    try:
                        # Load image
                        image = face_recognition.load_image_file(str(img_file))
                        
                        # Get face encoding
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            person_encodings.append(encodings[0])
                            successful_loads += 1
                            logger.debug(f"[FaceProcessor] Loaded face from: {img_file.name}")
                        else:
                            logger.warning(f"[FaceProcessor] No face found in: {img_file.name}")
                            
                    except Exception as e:
                        logger.error(f"[FaceProcessor] Error loading {img_file.name}: {e}")
                
                # If we got encodings for this person, add them
                if person_encodings:
                    # Use the average encoding for this person (or just the first one)
                    avg_encoding = np.mean(person_encodings, axis=0)
                    self.known_face_encodings.append(avg_encoding)
                    self.known_face_names.append(person_name)
                    logger.info(f"[FaceProcessor] Added {person_name} with {successful_loads} face samples")
                else:
                    logger.warning(f"[FaceProcessor] No valid faces found for {person_name}")
        
        # Cache the encodings for faster startup next time
        if self.known_face_encodings:
            try:
                with open(encodings_cache, 'wb') as f:
                    pickle.dump({
                        'encodings': self.known_face_encodings,
                        'names': self.known_face_names
                    }, f)
                logger.info(f"[FaceProcessor] Cached encodings saved")
            except Exception as e:
                logger.error(f"[FaceProcessor] Cache save failed: {e}")
    
    def process_frame(self, frame: np.ndarray):
        """
        Detect and recognize faces in frame.
        Returns list of dicts with 'box', 'name', and 'confidence' keys.
        """
        if frame is None or len(self.known_face_encodings) == 0:
            return []
        
        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame for faster processing
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        #face_locations = face_recognition.face_locations(small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        face_data = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back to original size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            name = "Unknown"
            confidence = 0.0
            
            if self.known_face_encodings:
                # Compare with known faces
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                # Consider it a match if distance is less than 0.6 (stricter threshold)
                if face_distances[best_match_index] < 0.8:  # More strict for better accuracy
                    name = self.known_face_names[best_match_index]
                    confidence = 1.0 - face_distances[best_match_index]
            
            face_data.append({
                "box": [left, top, right, bottom],
                "name": name,
                "confidence": float(confidence)
            })
        
        return face_data
    
    def refresh_cache(self):
        """Refresh the face encodings cache"""
        if FACE_ENCODINGS_CACHE.exists():
            FACE_ENCODINGS_CACHE.unlink()
        self._load_known_faces_from_person_folders()
        return len(self.known_face_names)
    
    def get_known_faces_info(self):
        """Get information about loaded faces"""
        return {
            "total_faces": len(self.known_face_names),
            "names": self.known_face_names,
            "fase_data_path": str(self.fase_data_path),
            "cache_path": str(self.cache_path),
            "method": "face_recognition (HOG)",
            "key_landmarks_count": 128,  # dlib's 128-d embeddings
        }
