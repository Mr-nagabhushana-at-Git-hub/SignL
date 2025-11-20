# src/majorSignL/models/mediapipe_face_recognizer.py

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import pickle
import logging
from typing import List, Dict, Tuple, Optional

# Try sklearn cosine_similarity, fallback to a simple implementation if unavailable
try:
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity  # type: ignore

    def cosine_similarity(A, B):
        return _sk_cosine_similarity(A, B)
except Exception:
    def cosine_similarity(A, B):
        """Fallback cosine similarity: A shape (1, d), B shape (n, d) -> (1, n)"""
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        if A.ndim == 1:
            A = A[None, :]
        # Normalize
        def _norm(x):
            n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
            return x / n
        A_n = _norm(A)
        B_n = _norm(B)
        sim = A_n @ B_n.T
        return sim

logger = logging.getLogger(__name__)

class MediaPipeFaceRecognizer:
    """
    Face recognition using MediaPipe face landmarks instead of traditional face encodings.
    Extracts 468 face landmarks and uses geometric features for comparison.
    """
    
    def __init__(self, data_path: Path):
        """Initialize enhanced MediaPipe face recognition system"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.fase_data_path = self.project_root / "src" / "data" / "fase_data"
        self.cache_path = self.project_root / "src" / "data" / "face_cache"
        
        # Create cache directory
        self.cache_path.mkdir(exist_ok=True)
        
        # Initialize MediaPipe Face Detection (for better face localization in full body photos)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range detection (better for full body photos)
            min_detection_confidence=0.3  # Lower threshold for challenging conditions
        )
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lower for better detection
            min_tracking_confidence=0.3
        )
        
        # Enhanced key landmark indices for better recognition
        self.key_landmarks = [
            # Face contour (comprehensive outline)
            10, 151, 9, 8, 168, 6, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 
            21, 54, 103, 67, 109, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
            
            # Eyes (comprehensive)
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 
            362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
            
            # Eyebrows
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            296, 334, 293, 300, 276, 283, 282, 295, 285, 336,
            
            # Nose (detailed)
            1, 2, 5, 4, 6, 19, 94, 125, 141, 235, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122, 
            51, 48, 115, 131, 134, 102, 49, 220, 305, 307, 375, 321, 308, 324, 318,
            
            # Mouth (comprehensive)
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            
            # Cheeks and jaw (enhanced)
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 
            116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187
        ]
        
        # Remove duplicates and sort for consistency
        self.key_landmarks = sorted(list(set(self.key_landmarks)))
        
        # Storage for known face features
        self.known_face_features = []
        self.known_face_names = []
        
        # Additional pre-trained model embeddings
        self.pretrained_embeddings = {
            'facenet': [],
            'vggface': [],
            'vgg16': []
        }
        self.pretrained_names = {
            'facenet': [],
            'vggface': [],
            'vgg16': []
        }
        
        # Enhanced similarity threshold - increased to reduce false positives
        self.similarity_threshold = 0.85  # Higher threshold for more accurate matching

        # Add PyTorch classifier for better accuracy
        self.classifier_model = None
        self._build_classifier()

        # Throttle logging of matches to avoid spam
        self._last_logged_name = None
        self._last_logged_conf = 0.0
        self._last_log_ts = 0.0

        # Temporal stabilization state
        self._stable_name = None
        self._stable_conf = 0.0
        self._stable_count = 0
        self._stable_bbox = None
        self._required_consecutive = 3  # require N consecutive frames to confirm identity

        # Identity hysteresis: confirmed identity stickiness and cooldown to prevent drift
        self._confirmed_name = None
        self._confirmed_conf = 0.0
        self._confirmed_bbox = None
        self._confirmed_since_ts = 0.0
        self._drop_below_count = 0
        # Thresholds - adjusted for MediaPipe landmark features (typically 0.4-0.7 range)
        self._confirm_sim = 0.45  # Lower threshold for MediaPipe features
        self._sticky_sim = 0.40   # Lower sticky threshold
        self._drop_sim = 0.35    # Lower drop threshold
        self._margin_min = 0.02  # Smaller margin requirement
        self._switch_cooldown_s = 1.2

        # Gating thresholds
        self._iou_threshold = 0.3

        # Optional OpenCV DNN face detector (res10 SSD) for robust face gating
        self.dnn_net = None
        self._dnn_enabled = False
        self._dnn_fail_count = 0
        try:
            model_dir = self.project_root / "src" / "data" / "models"
            proto = model_dir / "deploy.prototxt"
            caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            if proto.exists() and caffemodel.exists():
                net = cv2.dnn.readNetFromCaffe(str(proto), str(caffemodel))
                self.dnn_net = net
                self._dnn_enabled = True
                logger.info("[MediaPipeFaceRecognizer] OpenCV DNN face detector enabled")
        except Exception as _e:
            self.dnn_net = None
            self._dnn_enabled = False
            logger.warning(f"[MediaPipeFaceRecognizer] DNN load failed: {_e}")
        
        # Load known faces
        self._load_known_faces_from_folders()
        
        # Load pre-trained embeddings from .pkl files
        self._load_pretrained_embeddings()
        
        # Try to load existing classifier, otherwise build new one
        self.classifier_model = self._load_classifier()
        if self.classifier_model is None:
            self.classifier_model = self._build_classifier()
            if self.classifier_model is not None:
                self._save_classifier()
        
        logger.info(f"[MediaPipeFaceRecognizer] Loaded {len(self.known_face_names)} known faces")
        print(f"[MediaPipeFaceRecognizer] Loaded {len(self.known_face_names)} known faces")
    
    def _build_classifier(self):
        """Build PyTorch classifier for face recognition"""
        if len(self.known_face_features) == 0:
            logger.warning("[MediaPipeFaceRecognizer] No training data available for classifier")
            return None
            
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            
            # Define neural network architecture
            class FaceClassifier(nn.Module):
                def __init__(self, input_size, num_classes):
                    super(FaceClassifier, self).__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, num_classes)
                    )
                    
                def forward(self, x):
                    return self.layers(x)
            
            # Create model
            input_size = len(self.known_face_features[0])
            num_classes = len(self.known_face_names)
            model = FaceClassifier(input_size, num_classes)
            
            # Prepare training data
            features_tensor = torch.FloatTensor(self.known_face_features)
            labels_tensor = torch.LongTensor(range(num_classes))
            
            # Create data loader for batching
            dataset = TensorDataset(features_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train for a few epochs
            model.train()
            for epoch in range(50):  # Quick training
                for batch_features, batch_labels in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                if epoch % 10 == 0:
                    logger.debug(f"[MediaPipeFaceRecognizer] Training epoch {epoch}, loss: {loss.item():.4f}")
            
            logger.info(f"[MediaPipeFaceRecognizer] Classifier trained with {num_classes} classes")
            return model
            
        except ImportError as e:
            logger.warning(f"[MediaPipeFaceRecognizer] PyTorch not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[MediaPipeFaceRecognizer] Classifier training failed: {e}")
            return None
    
    def _save_classifier(self):
        """Save trained classifier to disk"""
        if self.classifier_model is None:
            return
            
        try:
            import torch
            model_path = self.project_root / "src" / "data" / "models" / "face_classifier.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.classifier_model.state_dict(),
                'num_classes': len(self.known_face_names),
                'class_names': self.known_face_names
            }, model_path)
            
            logger.info(f"[MediaPipeFaceRecognizer] Saved classifier to {model_path}")
            
        except Exception as e:
            logger.warning(f"[MediaPipeFaceRecognizer] Could not save classifier: {e}")
    
    def _load_classifier(self):
        """Load pre-trained classifier from disk"""
        try:
            import torch
            model_path = self.project_root / "src" / "data" / "models" / "face_classifier.pth"
            
            if not model_path.exists():
                return None
                
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Rebuild model with correct architecture
            input_size = len(self.known_face_features[0]) if len(self.known_face_features) > 0 else 500
            num_classes = len(self.known_face_names)
            
            class FaceClassifier(torch.nn.Module):
                def __init__(self, input_size, num_classes):
                    super(FaceClassifier, self).__init__()
                    self.layers = torch.nn.Sequential(
                        torch.nn.Linear(input_size, 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(256, 128),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, num_classes)
                    )
                    
                def forward(self, x):
                    return self.layers(x)
            
            model = FaceClassifier(input_size, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"[MediaPipeFaceRecognizer] Loaded classifier from {model_path}")
            return model
            
        except Exception as e:
            logger.warning(f"[MediaPipeFaceRecognizer] Could not load classifier: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better face detection in challenging conditions"""
        try:
            # Enhance contrast and brightness for better detection
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def _detect_and_crop_face(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect face and return cropped face image with bounding box for full body photos"""
        try:
            # Preprocess image for better detection
            enhanced_image = self._preprocess_image(image)
            rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            
            h, w = image.shape[:2]

            # Prefer OpenCV DNN detection if available; fallback to MediaPipe face_detection
            dnn_bbox = self._detect_face_dnn(enhanced_image)
            if dnn_bbox is not None:
                x1, y1, x2, y2 = dnn_bbox
            else:
                results = self.face_detection.process(rgb_image)
                if not results.detections:
                    return None
                detection = results.detections[0]
                rb = detection.location_data.relative_bounding_box
                padding = 30
                x = max(0, int(rb.xmin * w) - padding)
                y = max(0, int(rb.ymin * h) - padding)
                width = min(w - x, int(rb.width * w) + 2 * padding)
                height = min(h - y, int(rb.height * h) + 2 * padding)
                x1, y1, x2, y2 = x, y, x + width, y + height
            
            # Crop face region
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0 or min(face_crop.shape[:2]) < 50:
                return None
                
            return face_crop, (x1, y1, x2, y2)
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None
    
    def _extract_face_features(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]]:
        """
        Enhanced face feature extraction with better preprocessing and face detection
        Handles full body photos and challenging backgrounds
        Returns a tuple of (features, bounding_box)
        """
        try:
            # First attempt: detect and crop face for better landmark detection
            face_result = self._detect_and_crop_face(image)

            if face_result is not None:
                face_image, bbox = face_result
            else:
                # No face detected - return None to indicate no face found
                return None, None

            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe Face Mesh
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return None, None

            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]

            # Extract enhanced landmarks
            h, w = face_image.shape[:2]
            landmarks_3d = []

            # Extract all key landmarks with 3D coordinates
            for idx in self.key_landmarks:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    # Normalize coordinates to [0,1] range
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z if hasattr(landmark, 'z') else 0
                    landmarks_3d.extend([x, y, z])

            if not landmarks_3d:
                return None, None

            # Convert to numpy array
            landmark_features = np.array(landmarks_3d)

            # Calculate enhanced geometric features
            geometric_features = self._calculate_enhanced_geometric_features(face_landmarks, w, h)

            # Calculate facial proportions
            proportion_features = self._calculate_facial_proportions(face_landmarks)

            # Combine all features
            combined_features = np.concatenate([
                landmark_features,
                geometric_features,
                proportion_features
            ])

            # Normalize the final feature vector
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm

            return combined_features, bbox

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None, None
    
    def _calculate_enhanced_geometric_features(self, face_landmarks, w: int, h: int) -> np.ndarray:
        """Calculate enhanced geometric features for better face recognition"""
        try:
            landmarks = face_landmarks.landmark
            
            # Eye measurements
            left_eye_inner = landmarks[133]
            left_eye_outer = landmarks[33]
            right_eye_inner = landmarks[362]
            right_eye_outer = landmarks[263]
            
            # Inter-eye distance
            eye_distance = np.sqrt((left_eye_inner.x - right_eye_inner.x)**2 + (left_eye_inner.y - right_eye_inner.y)**2)
            
            # Eye widths
            left_eye_width = abs(left_eye_outer.x - left_eye_inner.x)
            right_eye_width = abs(right_eye_outer.x - right_eye_inner.x)
            
            # Nose measurements
            nose_tip = landmarks[1]
            nose_bridge = landmarks[6]
            nose_left = landmarks[31]
            nose_right = landmarks[235]
            
            nose_width = abs(nose_right.x - nose_left.x)
            nose_height = abs(nose_tip.y - nose_bridge.y)
            
            # Mouth measurements
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            mouth_top = landmarks[13]
            mouth_bottom = landmarks[14]
            
            mouth_width = abs(mouth_right.x - mouth_left.x)
            mouth_height = abs(mouth_bottom.y - mouth_top.y)
            
            # Face dimensions
            face_left = landmarks[172]
            face_right = landmarks[397]
            face_top = landmarks[10]
            face_bottom = landmarks[152]
            
            face_width = abs(face_right.x - face_left.x)
            face_height = abs(face_bottom.y - face_top.y)
            face_ratio = face_width / face_height if face_height > 0 else 0
            
            # Eyebrow positions
            left_eyebrow_inner = landmarks[70]
            left_eyebrow_outer = landmarks[46]
            right_eyebrow_inner = landmarks[296]
            right_eyebrow_outer = landmarks[285]
            
            # Cheek measurements
            left_cheek = landmarks[116]
            right_cheek = landmarks[345]
            cheek_distance = np.sqrt((left_cheek.x - right_cheek.x)**2 + (left_cheek.y - right_cheek.y)**2)
            
            # Jaw measurements
            jaw_left = landmarks[172]
            jaw_right = landmarks[397]
            chin = landmarks[175]
            
            jaw_width = abs(jaw_right.x - jaw_left.x)
            
            # Additional ratios for better discrimination
            eye_nose_ratio = eye_distance / nose_width if nose_width > 0 else 0
            nose_mouth_ratio = nose_width / mouth_width if mouth_width > 0 else 0
            eye_mouth_ratio = eye_distance / mouth_width if mouth_width > 0 else 0
            
            geometric_features = np.array([
                eye_distance, left_eye_width, right_eye_width,
                nose_width, nose_height, mouth_width, mouth_height,
                face_width, face_height, face_ratio,
                cheek_distance, jaw_width,
                eye_nose_ratio, nose_mouth_ratio, eye_mouth_ratio
            ])
            
            return geometric_features
            
        except Exception as e:
            logger.error(f"Enhanced geometric feature calculation error: {e}")
            return np.zeros(15)
    
    def _calculate_facial_proportions(self, face_landmarks) -> np.ndarray:
        """Calculate facial proportions for enhanced recognition"""
        try:
            landmarks = face_landmarks.landmark
            
            # Golden ratio proportions
            face_top = landmarks[10]
            face_bottom = landmarks[152]
            eye_level = landmarks[168]
            nose_bottom = landmarks[2]
            
            total_face_height = abs(face_bottom.y - face_top.y)
            
            if total_face_height == 0:
                return np.zeros(8)
            
            # Upper face proportion (forehead to eyes)
            upper_face = abs(eye_level.y - face_top.y) / total_face_height
            
            # Middle face proportion (eyes to nose)
            middle_face = abs(nose_bottom.y - eye_level.y) / total_face_height
            
            # Lower face proportion (nose to chin)
            lower_face = abs(face_bottom.y - nose_bottom.y) / total_face_height
            
            # Face width at different levels
            face_left = landmarks[172]
            face_right = landmarks[397]
            face_width = abs(face_right.x - face_left.x)
            
            eye_left = landmarks[33]
            eye_right = landmarks[263]
            eye_width = abs(eye_right.x - eye_left.x)
            
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            mouth_level_width = abs(mouth_right.x - mouth_left.x)
            
            # Width ratios
            eye_to_face_ratio = eye_width / face_width if face_width > 0 else 0
            mouth_to_face_ratio = mouth_level_width / face_width if face_width > 0 else 0
            
            # Symmetry measures
            face_center_x = (face_left.x + face_right.x) / 2
            nose_center_x = landmarks[1].x
            symmetry = abs(nose_center_x - face_center_x)
            
            proportion_features = np.array([
                upper_face, middle_face, lower_face,
                eye_to_face_ratio, mouth_to_face_ratio,
                symmetry, face_width, total_face_height
            ])
            
            return proportion_features
            
        except Exception as e:
            logger.error(f"Facial proportion calculation error: {e}")
            return np.zeros(8)
    
    def _load_known_faces_from_folders(self):
        """Load face features from person folders"""
        cache_file = self.cache_path / "mediapipe_face_features.pkl"
        
        # Try loading cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.known_face_features = cached_data['features']
                    self.known_face_names = cached_data['names']
                    logger.info(f"[MediaPipeFaceRecognizer] Loaded cached features for {len(self.known_face_names)} faces")
                    print(f"[MediaPipeFaceRecognizer] Loaded cached features for {len(self.known_face_names)} faces")
                    return
            except Exception as e:
                logger.warning(f"[MediaPipeFaceRecognizer] Cache load failed: {e}")
        
        # Load from person folders
        if not self.fase_data_path.exists():
            logger.error(f"[MediaPipeFaceRecognizer] fase_data path not found: {self.fase_data_path}")
            return
        
        for person_folder in self.fase_data_path.iterdir():
            if person_folder.is_dir() and not person_folder.name.endswith('.pkl'):
                person_name = person_folder.name
                logger.info(f"[MediaPipeFaceRecognizer] Processing person: {person_name}")
                print(f"[MediaPipeFaceRecognizer] Processing person: {person_name}")
                
                # Get all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(person_folder.glob(ext))
                
                person_features = []
                successful_loads = 0
                total_files = len(image_files)
                
                logger.info(f"[MediaPipeFaceRecognizer] Processing {total_files} images for {person_name}")
                print(f"[MediaPipeFaceRecognizer] Processing {total_files} images for {person_name}")
                
                # Process all images for this person with enhanced detection
                for img_file in image_files:
                    try:
                        # Load image
                        image = cv2.imread(str(img_file))
                        if image is None:
                            logger.warning(f"[MediaPipeFaceRecognizer] Could not load: {img_file.name}")
                            continue
                        
                        # Resize if image is too large for faster processing
                        h, w = image.shape[:2]
                        if max(h, w) > 1024:
                            scale = 1024 / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            image = cv2.resize(image, (new_w, new_h))
                        
                        # Extract face features with enhanced detection
                        features, _ = self._extract_face_features(image)
                        
                        if features is not None:
                            person_features.append(features)
                            successful_loads += 1
                            logger.debug(f"[MediaPipeFaceRecognizer] ✅ Loaded features from: {img_file.name}")
                        else:
                            logger.warning(f"[MediaPipeFaceRecognizer] ❌ No face found in: {img_file.name}")
                            
                    except Exception as e:
                        logger.error(f"[MediaPipeFaceRecognizer] Error loading {img_file.name}: {e}")
                
                # If we got features for this person, add them
                if person_features:
                    # Use average features for this person (more robust than single image)
                    avg_features = np.mean(person_features, axis=0)
                    self.known_face_features.append(avg_features)
                    self.known_face_names.append(person_name)
                    
                    success_rate = (successful_loads / total_files) * 100 if total_files > 0 else 0
                    logger.info(f"[MediaPipeFaceRecognizer] ✅ Added {person_name} with {successful_loads}/{total_files} samples ({success_rate:.1f}% success)")
                    print(f"[MediaPipeFaceRecognizer] ✅ Added {person_name} with {successful_loads}/{total_files} samples ({success_rate:.1f}% success)")
                else:
                    logger.warning(f"[MediaPipeFaceRecognizer] ❌ No valid faces found for {person_name}")
                    print(f"[MediaPipeFaceRecognizer] ❌ No valid faces found for {person_name}")
        
        # Cache the features
        if self.known_face_features:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'features': self.known_face_features,
                        'names': self.known_face_names
                    }, f)
                logger.info(f"[MediaPipeFaceRecognizer] Cached features saved")
                print(f"[MediaPipeFaceRecognizer] Cached features saved")
            except Exception as e:
                logger.error(f"[MediaPipeFaceRecognizer] Cache save failed: {e}")
    
    def _load_pretrained_embeddings(self):
        """Load pre-trained embeddings from .pkl files for additional verification"""
        import pickle
        
        pkl_files = {
            'facenet': 'ds_model_facenet_detector_opencv_aligned_normalization_base_expand_0.pkl',
            'vggface': 'ds_model_vggface_detector_opencv_aligned_normalization_base_expand_0.pkl',
            'vgg16': 'ds_model_vgg16_detector_opencv_aligned_normalization_base_expand_0.pkl'
        }
        
        for model_name, filename in pkl_files.items():
            pkl_path = self.fase_data_path / filename
            if pkl_path.exists():
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, list) and len(data) > 0:
                        embeddings = []
                        names = []
                        for item in data:
                            if isinstance(item, dict) and 'embedding' in item and 'identity' in item:
                                emb = np.array(item['embedding'])
                                if emb.shape[0] > 0:  # Ensure valid embedding
                                    embeddings.append(emb)
                                    # Extract person name from path
                                    identity_path = item['identity']
                                    person_name = Path(identity_path).parent.name
                                    names.append(person_name)
                        
                        if embeddings:
                            self.pretrained_embeddings[model_name] = np.array(embeddings)
                            self.pretrained_names[model_name] = names
                            logger.info(f"[MediaPipeFaceRecognizer] Loaded {len(embeddings)} {model_name} embeddings")
                        else:
                            logger.warning(f"[MediaPipeFaceRecognizer] No valid embeddings in {filename}")
                    else:
                        logger.warning(f"[MediaPipeFaceRecognizer] {filename} is empty or invalid")
                        
                except Exception as e:
                    logger.error(f"[MediaPipeFaceRecognizer] Failed to load {filename}: {e}")
            else:
                logger.info(f"[MediaPipeFaceRecognizer] {filename} not found, skipping {model_name}")
    
    def _get_pretrained_similarity(self, query_embedding, model_name):
        """Get similarity score from pre-trained model embeddings"""
        if model_name not in self.pretrained_embeddings or len(self.pretrained_embeddings[model_name]) == 0:
            return 0.0, None
        
        embeddings = self.pretrained_embeddings[model_name]
        names = self.pretrained_names[model_name]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_sim = float(similarities[best_idx])
        best_name = names[best_idx]
        
        return best_sim, best_name
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Enhanced face detection and recognition in frame using MediaPipe landmarks
        Returns list of dicts with 'box', 'name', and 'confidence' keys
        """
        if frame is None:
            return []
            
        if not self.known_face_features:
            logger.warning("[MediaPipeFaceRecognizer] No known faces loaded!")
            return []

        try:
            # Extract features and bounding box from the current frame in one step
            frame_features, bbox = self._extract_face_features(frame)

            # If no face features detected, return empty list (no hallucination)
            if frame_features is None or bbox is None:
                return []

            # Find best match among known faces
            best_match_name = "Unknown"
            best_confidence = 0.0

            # Calculate cosine similarities with all known faces in a single operation
            similarities = cosine_similarity([frame_features], self.known_face_features)[0]

            # Debug: log similarity values (only occasionally to avoid spam)
            import time
            current_time = time.time()
            if not hasattr(self, '_last_debug_time'):
                self._last_debug_time = 0
            
            if current_time - self._last_debug_time > 5:  # Log every 5 seconds
                logger.info(f"[MediaPipeFaceRecognizer] Similarities range: {np.min(similarities):.3f} - {np.max(similarities):.3f}")
                logger.info(f"[MediaPipeFaceRecognizer] Best similarity: {np.max(similarities):.3f} at index {np.argmax(similarities)}")
                logger.info(f"[MediaPipeFaceRecognizer] Name at best index: {self.known_face_names[np.argmax(similarities)]}")
                logger.info(f"[MediaPipeFaceRecognizer] Thresholds: confirm={self._confirm_sim:.3f}, sticky={self._sticky_sim:.3f}")
                self._last_debug_time = current_time

            # Use PyTorch classifier if available for better accuracy
            classifier_prediction = None
            if self.classifier_model is not None and len(self.known_face_names) > 0:
                try:
                    import torch
                    self.classifier_model.eval()
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(frame_features).unsqueeze(0)
                        outputs = self.classifier_model(features_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        pred_class = torch.argmax(outputs, dim=1).item()
                        pred_prob = probabilities[0][pred_class].item()
                        
                        if pred_class < len(self.known_face_names):
                            classifier_prediction = {
                                'name': self.known_face_names[pred_class],
                                'confidence': pred_prob
                            }
                            logger.debug(f"[MediaPipeFaceRecognizer] Classifier prediction: {classifier_prediction}")
                except Exception as e:
                    logger.warning(f"[MediaPipeFaceRecognizer] Classifier prediction failed: {e}")

            # Sort to get top-2 for margin
            order = np.argsort(similarities)[::-1]
            best_match_idx = int(order[0])
            best_similarity = float(similarities[best_match_idx])
            second_best = float(similarities[order[1]]) if len(order) > 1 else 0.0
            margin = best_similarity - second_best

            # Enhanced candidate selection with classifier verification
            candidate_name = None
            candidate_conf = 0.0
            
            # Primary: similarity-based matching
            if best_similarity >= self._confirm_sim:
                similarity_name = self.known_face_names[best_match_idx]
                similarity_conf = best_similarity
                
                # Secondary: classifier verification
                if classifier_prediction and classifier_prediction['confidence'] > 0.7:
                    if classifier_prediction['name'] == similarity_name:
                        # Agreement between methods - boost confidence
                        candidate_name = similarity_name
                        candidate_conf = min(1.0, (similarity_conf + classifier_prediction['confidence']) / 2 + 0.1)
                        logger.debug(f"[MediaPipeFaceRecognizer] Methods agree: {candidate_name}")
                    else:
                        # Disagreement - use similarity but lower confidence
                        candidate_name = similarity_name
                        candidate_conf = similarity_conf * 0.8
                        logger.debug(f"[MediaPipeFaceRecognizer] Methods disagree: sim={similarity_name}, clf={classifier_prediction['name']}")
                else:
                    # No classifier or low confidence - use similarity
                    candidate_name = similarity_name
                    candidate_conf = similarity_conf
                    
                logger.debug(f"[MediaPipeFaceRecognizer] Candidate: {candidate_name} with sim {best_similarity:.3f}, margin {margin:.3f}")
                
                # TODO: Re-enable pre-trained model integration with proper feature extraction
                # For now, focus on MediaPipe features only

            # IoU gating for track consistency
            iou_ok = True
            if self._stable_bbox is not None and bbox is not None:
                iou_ok = self._bbox_iou(self._stable_bbox, bbox) > self._iou_threshold

            # If we have a confirmed identity, keep it sticky unless strong evidence to switch
            import time as _time
            now = _time.time()

            if self._confirmed_name is not None and iou_ok:
                # Similarity to confirmed identity
                try:
                    confirmed_idx = self.known_face_names.index(self._confirmed_name)
                    sim_confirmed = float(similarities[confirmed_idx])
                except ValueError:
                    sim_confirmed = 0.0

                if sim_confirmed >= self._sticky_sim:
                    # Stay with confirmed and reset drop counter
                    self._drop_below_count = 0
                    best_match_name = self._confirmed_name
                    best_confidence = max(self._confirmed_conf, sim_confirmed)
                else:
                    # Potential drop; allow some consecutive drops before switching
                    self._drop_below_count += 1
                    # Consider switching only after cooldown and enough drops and strong candidate
                    allow_switch = (
                        (now - self._confirmed_since_ts) >= self._switch_cooldown_s and
                        self._drop_below_count >= max(3, self._required_consecutive) and
                        candidate_name is not None and candidate_name != self._confirmed_name
                    )

                    if allow_switch:
                        # Use stabilization for candidate
                        if self._stable_name == candidate_name and iou_ok:
                            self._stable_count += 1
                            self._stable_conf = max(self._stable_conf, candidate_conf)
                        else:
                            self._stable_name = candidate_name
                            self._stable_conf = candidate_conf
                            self._stable_count = 1
                            self._stable_bbox = bbox

                        if self._stable_count >= self._required_consecutive:
                            # Confirm switch
                            self._confirmed_name = self._stable_name
                            self._confirmed_conf = self._stable_conf
                            self._confirmed_bbox = self._stable_bbox
                            self._confirmed_since_ts = now
                            self._drop_below_count = 0
                            best_match_name = self._confirmed_name
                            best_confidence = self._confirmed_conf
                        else:
                            # Until confirmed, keep previous confirmed to avoid flicker
                            best_match_name = self._confirmed_name
                            best_confidence = max(self._confirmed_conf, sim_confirmed)
                    else:
                        # Keep confirmed during cooldown/drops
                        best_match_name = self._confirmed_name
                        best_confidence = max(self._confirmed_conf, sim_confirmed)
            else:
                # No confirmed identity yet: use stabilization to confirm candidate
                if candidate_name is not None and iou_ok:
                    if self._stable_name == candidate_name:
                        self._stable_count += 1
                        self._stable_conf = max(self._stable_conf, candidate_conf)
                    else:
                        self._stable_name = candidate_name
                        self._stable_conf = candidate_conf
                        self._stable_count = 1
                        self._stable_bbox = bbox

                    if self._stable_count >= self._required_consecutive:
                        self._confirmed_name = self._stable_name
                        self._confirmed_conf = self._stable_conf
                        self._confirmed_bbox = self._stable_bbox
                        self._confirmed_since_ts = now
                        self._drop_below_count = 0
                        best_match_name = self._confirmed_name
                        best_confidence = self._confirmed_conf
                else:
                    # No strong candidate; remain Unknown
                    self._stable_count = max(0, self._stable_count - 1)
                    if self._stable_count == 0:
                        self._stable_name = None
                        self._stable_conf = 0.0
                        self._stable_bbox = None

            # Use the bounding box returned by the feature extraction
            # No fallback to full frame - if we don't have a proper bbox, we don't have a face
            face_box = list(bbox)

            return [{
                "box": face_box,
                "name": best_match_name,
                "confidence": best_confidence
            }]

        except Exception as e:
            logger.error(f"Face processing error in process_frame: {e}")
            return []

    def force_retrain_all_faces(self) -> Dict:
        """
        Force a full retrain of all faces from the data folders, clearing the cache.
        This is useful when new images are added or model parameters change.
        """
        cache_file = self.cache_path / "mediapipe_face_features.pkl"
        
        # Clear existing features
        self.known_face_features = []
        self.known_face_names = []
        
        # Delete cache file
        if cache_file.exists():
            try:
                cache_file.unlink()
                logger.info("[MediaPipeFaceRecognizer] Deleted existing cache file.")
            except Exception as e:
                logger.error(f"[MediaPipeFaceRecognizer] Could not delete cache file: {e}")
        
        # Reload from folders
        self._load_known_faces_from_folders()
        
        # Rebuild PyTorch classifier with new data
        self.classifier_model = self._build_classifier()
        if self.classifier_model is not None:
            self._save_classifier()
        
        # Return status
        return {
            "total_faces": len(self.known_face_names),
            "names": self.known_face_names,
            "message": "Forced retraining complete."
        }
    
    def get_known_faces_info(self) -> Dict:
        """Get information about the loaded face recognition model"""
        return {
            "method": "MediaPipe Face Mesh",
            "total_faces": len(self.known_face_names),
            "names": self.known_face_names,
            "fase_data_path": str(self.fase_data_path),
            "cache_path": str(self.cache_path / "mediapipe_face_features.pkl"),
            "key_landmarks_count": len(self.key_landmarks),
            "feature_vector_size": len(self.known_face_features[0]) if self.known_face_features else 0,
            "params": {
                "confirm_sim": self._confirm_sim,
                "sticky_sim": self._sticky_sim,
                "drop_sim": self._drop_sim,
                "margin_min": self._margin_min,
                "switch_cooldown_s": self._switch_cooldown_s,
                "required_consecutive": self._required_consecutive,
                "iou_threshold": self._iou_threshold
            }
        }

    def update_parameters(self, params: Dict) -> Dict:
        """Update recognition thresholds and return current values"""
        self._confirm_sim = float(params.get("confirm_sim", self._confirm_sim))
        self._sticky_sim = float(params.get("sticky_sim", self._sticky_sim))
        self._drop_sim = float(params.get("drop_sim", self._drop_sim))
        self._margin_min = float(params.get("margin_min", self._margin_min))
        self._switch_cooldown_s = float(params.get("switch_cooldown_s", self._switch_cooldown_s))
        self._required_consecutive = int(params.get("required_consecutive", self._required_consecutive))
        self._iou_threshold = float(params.get("iou_threshold", self._iou_threshold))
        return self.get_known_faces_info().get("params", {})
    
    def refresh_cache(self) -> int:
        """Alias for force_retrain_all_faces for compatibility"""
        result = self.force_retrain_all_faces()
        return result['total_faces']

    def _detect_face_dnn(self, image: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        """Detect face using OpenCV DNN (SSD) if available; returns bbox (x1,y1,x2,y2)"""
        try:
            if not getattr(self, '_dnn_enabled', False) or self.dnn_net is None:
                return None
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            if detections.shape[2] == 0:
                return None
            # Find top detection
            best_idx = None
            best_conf = 0.0
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > 0.6 and conf > best_conf:
                    best_conf = conf
                    best_idx = i
            if best_idx is None:
                return None
            box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            # Clamp to image
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
            # Expand a bit for landmark coverage
            pad_x = int(0.1 * (x2 - x1))
            pad_y = int(0.15 * (y2 - y1))
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x); y2 = min(h, y2 + int(0.8 * pad_y))
            return (x1, y1, x2, y2)
        except Exception as _e:
            # Auto-disable DNN after a failure to avoid repeated OpenCV spam
            try:
                self._dnn_fail_count = getattr(self, '_dnn_fail_count', 0) + 1
                if getattr(self, '_dnn_enabled', False):
                    logger.warning(f"[MediaPipeFaceRecognizer] DNN detect failed (disabling): {_e}")
                    self._dnn_enabled = False
            except Exception:
                pass
            return None

    @staticmethod
    def _bbox_iou(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> float:
        """Compute IoU between two bboxes (x1,y1,x2,y2)"""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        a1 = max(0, (b1[2]-b1[0])) * max(0, (b1[3]-b1[1]))
        a2 = max(0, (b2[2]-b2[0])) * max(0, (b2[3]-b2[1]))
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def close(self):
        """Clean up MediaPipe resources"""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            logger.info("[MediaPipeFaceRecognizer] Resources cleaned up")
        except Exception as e:
            logger.error(f"[MediaPipeFaceRecognizer] Cleanup error: {e}")
