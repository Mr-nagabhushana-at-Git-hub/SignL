#!/usr/bin/env python3
"""
PyTorch Face Recognition System
Uses deep learning embeddings for accurate face recognition
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2
from pathlib import Path
import pickle
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class PyTorchFaceEmbedder:
    """PyTorch-based face embedder using ResNet50"""

    def __init__(self, device: str = "cuda"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self.model: Optional[nn.Module] = None
        self.embedding_dim = 2048

        # Initialize model
        self._load_model()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        """Load pre-trained ResNet50 model"""
        try:
            # Load ResNet50 without classification head
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.fc = nn.Identity()  # Remove classification layer
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"✅ PyTorch Face Embedder loaded on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch model: {e}")
            self.model = None

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image"""
        if self.model is None:
            return None

        try:
            # Preprocess image
            if face_image.shape[2] == 3:  # BGR to RGB
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            tensor = self.transform(face_image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                embedding = self.model(tensor).cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"❌ Embedding extraction failed: {e}")
            return None

class PyTorchFaceRecognizer:
    """Complete PyTorch-based face recognition system"""

    def __init__(self, data_path: Path, device: str = "cuda"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device
        self.data_path = data_path
        self.fase_data_path = data_path.parent / "fase_data"
        self.cache_path = data_path.parent / "face_cache"

        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._dnn_disable_flag = self.cache_path / "disable_dnn_detector.flag"

        # Initialize embedder
        self.embedder = PyTorchFaceEmbedder(device=self.device)

        # Face detection (using OpenCV DNN)
        self.face_detector = None
        self.face_detector_type = "none"
        self._init_face_detector()

        # Recognition parameters
        self.similarity_threshold = 0.6  # Cosine similarity threshold
        self.max_distance = 0.8  # Maximum distance for matching

        # Storage for known faces
        self.embeddings = {}
        self.names = []

        # Load known faces
        self._load_known_faces()

        logger.info(f"🎯 PyTorch Face Recognizer initialized with {len(self.names)} known faces")

    def _init_face_detector(self):
        """Initialize OpenCV DNN face detector"""
        try:
            if self._dnn_disable_flag.exists():
                logger.info("⏭️ DNN disabled by flag; using Haar cascade detector")
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.face_detector_type = "haar"
                return

            model_dir = self.data_path
            proto = model_dir / "deploy.prototxt"
            caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

            if proto.exists() and caffemodel.exists():
                try:
                    self.face_detector = cv2.dnn.readNetFromCaffe(str(proto), str(caffemodel))
                    self.face_detector_type = "dnn"
                    # Sanity check: ensure the network has weights by running a dummy forward
                    try:
                        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
                        blob = cv2.dnn.blobFromImage(dummy, 1.0, (300, 300), (104.0, 177.0, 123.0))
                        self.face_detector.setInput(blob)
                        _ = self.face_detector.forward()
                        logger.info("✅ OpenCV DNN face detector loaded")
                    except cv2.error as forward_error:
                        logger.warning(
                            "⚠️ DNN model loaded but failed validation check (%s). This may indicate corrupted model files. Falling back to Haar cascade.",
                            forward_error,
                        )
                        try:
                            self._dnn_disable_flag.write_text(str(forward_error))
                        except OSError as flag_err:
                            logger.debug("Unable to write DNN disable flag: %s", flag_err)
                        self.face_detector = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        )
                        self.face_detector_type = "haar"
                except cv2.error as e:
                    logger.warning(f"⚠️ Failed to load DNN detector from {proto} and {caffemodel} ({e}). The files might be missing or corrupt. Falling back to Haar cascade.")
                    try:
                        self._dnn_disable_flag.write_text(str(e))
                    except OSError as flag_err:
                        logger.debug("Unable to write DNN disable flag: %s", flag_err)
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    self.face_detector_type = "haar"
            else:
                logger.warning(f"⚠️ DNN model files not found (checked for {proto} and {caffemodel}), using Haar cascades")
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.face_detector_type = "haar"
        except Exception as e:
            logger.error(f"❌ Face detector initialization failed: {e}")
            self.face_detector = None
            self.face_detector_type = "none"

    def get_known_faces_info(self) -> Dict[str, any]:
        """Returns a dictionary with information about the loaded faces."""
        return {
            "total_faces": len(self.names),
            "names": self.names,
            "fase_data_path": str(self.fase_data_path),
            "cache_path": str(self.cache_path),
            "method": "PyTorch ResNet50",
            "key_landmarks_count": self.embedding_dim,  # Using embedding dimension as a proxy
        }

    def _load_known_faces(self):
        """Load and cache face embeddings"""
        cache_file = self.cache_path / "pytorch_face_embeddings.pkl"

        # Try loading cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.embeddings = cached_data['embeddings']
                    self.names = cached_data['names']
                    logger.info(f"✅ Loaded cached embeddings for {len(self.names)} faces")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Cache load failed: {e}")

        # Load from person folders
        if not self.fase_data_path.exists():
            logger.error(f"❌ Face data path not found: {self.fase_data_path}")
            return

        logger.info("🔄 Computing embeddings from face images...")

        self.embeddings = {}
        self.names = []

        for person_folder in self.fase_data_path.iterdir():
            if person_folder.is_dir() and not person_folder.name.endswith('.pkl'):
                person_name = person_folder.name
                logger.info(f"👤 Processing {person_name}")

                # Get all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(person_folder.glob(ext))

                person_embeddings = []
                successful_loads = 0

                for img_file in image_files[:10]:  # Limit to 10 images per person
                    try:
                        image = cv2.imread(str(img_file))
                        if image is None:
                            continue

                        # Detect face
                        faces = self._detect_faces(image)
                        if len(faces) == 0:
                            continue

                        # Use first detected face
                        x, y, w, h = faces[0]
                        x = max(0, x)
                        y = max(0, y)
                        w = max(1, w)
                        h = max(1, h)
                        x2 = min(image.shape[1], x + w)
                        y2 = min(image.shape[0], y + h)

                        if x2 <= x or y2 <= y:
                            continue

                        face_crop = image[y:y2, x:x2]

                        # Extract embedding
                        embedding = self.embedder.extract_embedding(face_crop)
                        if embedding is not None:
                            person_embeddings.append(embedding)
                            successful_loads += 1

                    except Exception as e:
                        logger.error(f"❌ Error processing {img_file.name}: {e}")

                # Average embeddings for this person
                if person_embeddings:
                    avg_embedding = np.mean(person_embeddings, axis=0)
                    self.embeddings[person_name] = avg_embedding
                    self.names.append(person_name)

                    logger.info(f"✅ {person_name}: {successful_loads} embeddings averaged")

        # Cache the embeddings
        if self.embeddings:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'embeddings': self.embeddings,
                        'names': self.names
                    }, f)
                logger.info(f"💾 Cached embeddings saved")
            except Exception as e:
                logger.error(f"❌ Cache save failed: {e}")

    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image"""
        if self.face_detector is None:
            return []

        try:
            if hasattr(self.face_detector, 'detectMultiScale'):  # Haar cascade
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                return [(x, y, w, h) for (x, y, w, h) in faces]
            else:  # DNN detector
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
                try:
                    self.face_detector.setInput(blob)
                    detections = self.face_detector.forward()
                except cv2.error as e:
                    logger.warning(f"⚠️ DNN forward failed ({e}). Switching to Haar cascade.")
                    try:
                        self._dnn_disable_flag.write_text(str(e))
                    except OSError as flag_err:
                        logger.debug("Unable to write DNN disable flag: %s", flag_err)
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    self.face_detector_type = "haar"
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                    return [(x, y, w, h) for (x, y, w, h) in faces]

                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x, y, x2, y2 = box.astype("int")
                        x = max(0, x)
                        y = max(0, y)
                        x2 = min(w, x2)
                        y2 = min(h, y2)
                        if x2 > x and y2 > y:
                            faces.append((x, y, x2 - x, y2 - y))

                return faces

        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            return []

    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Recognize face from image"""
        if not self.embeddings:
            return "Unknown", 0.0

        # Extract embedding
        embedding = self.embedder.extract_embedding(face_image)
        if embedding is None:
            return "Unknown", 0.0

        # Find best match
        best_match = "Unknown"
        best_similarity = 0.0

        for name, known_embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(embedding, known_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        # Apply threshold
        if best_similarity < self.similarity_threshold:
            return "Unknown", best_similarity

        return best_match, best_similarity

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process frame for face recognition"""
        if frame is None:
            return []

        # Detect faces
        faces = self._detect_faces(frame)
        results = []

        height, width = frame.shape[:2]

        for (x, y, w, h) in faces:
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            if x2 <= x or y2 <= y:
                continue

            # Extract face crop
            face_crop = frame[y:y2, x:x2]

            # Recognize face
            name, confidence = self.recognize_face(face_crop)

            results.append({
                "box": [x, y, x2, y2],
                "name": name,
                "confidence": float(confidence)
            })

        return results

    def force_retrain_all_faces(self) -> Dict:
        """Force retrain all faces"""
        cache_file = self.cache_path / "pytorch_face_embeddings.pkl"
        if cache_file.exists():
            try:
                cache_file.unlink()
            except OSError as e:
                logger.warning(f"⚠️ Unable to remove cache file: {e}")

        self._load_known_faces()

        return {
            "total_faces": len(self.names),
            "names": self.names,
            "message": "PyTorch embeddings recomputed"
        }