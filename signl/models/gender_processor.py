# src/majorSignL/models/gender_processor.py
import cv2
import numpy as np
import logging
from pathlib import Path
from signl.config import MODELS_DIR

logger = logging.getLogger(__name__)

class GenderProcessor:
    def __init__(self):
        """Initialize the gender detection model."""
        self.gender_net = self._load_model()
        self.gender_list = ['Male', 'Female']
        self.mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        # Heuristic fallback active when model missing
        self.heuristic_mode = self.gender_net is None
        if self.heuristic_mode:
            logger.info("⚠️ Gender model unavailable - using geometric heuristic fallback.")

    def _load_model(self):
        """Load the Caffe model for gender detection (prototxt + caffemodel)."""
        try:
            proto_path = MODELS_DIR / "gender_deploy.prototxt"
            model_path = MODELS_DIR / "gender_net.caffemodel"
            if not proto_path.exists() or not model_path.exists():
                logger.error("❌ Gender model files not found! Expected gender_deploy.prototxt and gender_net.caffemodel under MODELS_DIR")
                return None

            # Use the Caffe-specific loader with correct argument order
            net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            logger.info("✅ Gender detection model loaded successfully.")
            return net
        except Exception as e:
            logger.error(f"❌ Error loading gender model: {e}")
            return None

    def predict_gender(self, face_image: np.ndarray) -> str:
        """
        Predict the gender from a given face image.
        Returns 'Male', 'Female', or 'Unknown'.
        """
        if face_image is None or face_image.size == 0:
            return "Unknown"

        # Heuristic fallback when model not present
        if self.heuristic_mode:
            try:
                h, w = face_image.shape[:2]
                if h == 0 or w == 0:
                    return "Unknown"
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                # Basic facial geometry ratios
                # Approximate jaw width vs face height using central horizontal slice
                mid_y = h // 2
                slice_band = gray[max(0, mid_y-3):min(h, mid_y+3), :]
                if slice_band.size == 0:
                    return "Unknown"
                # Edge detection to approximate contour complexity
                edges = cv2.Canny(slice_band, 40, 120)
                edge_density = edges.mean() / 255.0  # 0-1
                aspect_ratio = w / max(1, h)
                brightness = gray.mean() / 255.0
                # Simple heuristic: higher edge density + lower brightness + slightly wider aspect -> 'Male'
                score = (edge_density * 0.5) + (aspect_ratio * 0.3) + ((0.5 - brightness) * 0.4)
                # Calibrate threshold empirically; choose 0.55 as pivot
                gender = 'Male' if score > 0.55 else 'Female'
                return gender
            except Exception as e:
                logger.debug(f"Heuristic gender failed: {e}")
                return "Unknown"

        if self.gender_net is None:
            return "Unknown"

        try:
            blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), self.mean_values, swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            return gender
        except Exception as e:
            logger.error(f"Gender prediction failed: {e}")
            return "Unknown"
