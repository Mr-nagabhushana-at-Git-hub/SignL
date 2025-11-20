"""
Configuration module for SignL application
"""
from pathlib import Path
import logging

# Try to import torch, but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "signl" / "data"
FACE_DATA_DIR = DATA_DIR / "face_data"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DIR = DATA_DIR / "training"
CACHE_DIR = DATA_DIR / "cache"

# Frontend
FRONTEND_DIR = PROJECT_ROOT / "signl" / "frontend"

# Model paths
SIGN_LANGUAGE_MODEL = MODELS_DIR / "sign_language_transformer.pt"
FACE_ENCODINGS_CACHE = CACHE_DIR / "face_encodings.pkl"
MEDIAPIPE_FACE_FEATURES_CACHE = CACHE_DIR / "mediapipe_face_features.pkl"
PYTORCH_FACE_EMBEDDINGS_CACHE = CACHE_DIR / "pytorch_face_embeddings.pkl"

# Application settings
class AppConfig:
    """Application configuration"""
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = True
    
    # Device configuration
    DEVICE = "cuda" if (TORCH_AVAILABLE and torch and torch.cuda.is_available()) else "cpu"
    ENABLE_GPU = TORCH_AVAILABLE and torch and torch.cuda.is_available()
    
    # MediaPipe settings
    USE_GPU_MEDIAPIPE = True
    USE_FACE_MESH = True
    
    # Face recognition settings
    FACE_RECOGNITION_INTERVAL = 2  # Process every Nth frame
    FACE_CONFIDENCE_THRESHOLD = 0.6
    FACE_DETECTION_MODEL = "hog"  # "hog" or "cnn"
    
    # Sign language settings
    SIGN_CLASSIFICATION_INTERVAL = 3  # Process every Nth frame
    SIGN_CONFIDENCE_THRESHOLD = 0.7
    SIGN_SEQUENCE_LENGTH = 30
    
    # Performance settings
    JPEG_QUALITY = 90
    FRAME_RESIZE_FACTOR = 0.5
    FPS_LOG_INTERVAL = 30
    
    # Feature toggles
    ENABLE_FACE_RECOGNITION = True
    ENABLE_SIGN_LANGUAGE = True
    ENABLE_EMOTION_DETECTION = True
    ENABLE_GENDER_DETECTION = True

# Logging configuration
def setup_logging(level=logging.INFO):
    """Setup application logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()
