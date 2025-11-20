# src/majorSignL/models/gender_processor.py
import cv2
import numpy as np
import logging
from pathlib import Path
from majorSignL.config import MODELS_DIR

logger = logging.getLogger(__name__)

class GenderProcessor:
    def __init__(self):
        """Initialize the gender detection model."""
        self.gender_net = self._load_model()
        self.gender_list = ['Male', 'Female']
        self.mean_values = (78.4263377603, 87.7689143744, 114.895847746)

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
        if self.gender_net is None or face_image is None or face_image.size == 0:
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
