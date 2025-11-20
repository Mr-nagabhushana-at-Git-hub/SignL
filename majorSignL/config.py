# src/majorSignL/config.py
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = get_project_root()

# Data paths
DATA_DIR = PROJECT_ROOT / "src" / "data"
FACE_DATA_DIR = DATA_DIR / "fase_data"
FACE_CACHE_DIR = DATA_DIR / "face_cache"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DIR = DATA_DIR / "training"

# Model file paths
SIGN_LANGUAGE_MODEL = MODELS_DIR / "sign_language_transformer.pt"
FACE_ENCODINGS_CACHE = FACE_CACHE_DIR / "face_encodings.pkl"
MEDIAPIPE_FACE_FEATURES_CACHE = FACE_CACHE_DIR / "mediapipe_face_features.pkl"
PYTORCH_FACE_EMBEDDINGS_CACHE = FACE_CACHE_DIR / "pytorch_face_embeddings.pkl"

# Frontend path
FRONTEND_DIR = PROJECT_ROOT / "src" / "majorSignL" / "frontend"
