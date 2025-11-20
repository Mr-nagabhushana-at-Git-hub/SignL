#!/usr/bin/env python3
"""
Simple test script for face recognition
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from majorSignL.models.pytorch_face_recognizer import PyTorchFaceRecognizer

def test_face_recognition(device: str = "cuda"):
    """Test face recognition with a simple image"""
    print("🔍 Testing Face Recognition...")

    # Initialize face recognizer
    data_path = Path(__file__).parent / "src" / "data" / "models"
    face_recognizer = PyTorchFaceRecognizer(data_path, device=device)

    # Create a simple test image (you can replace with actual face image)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test face processing
    results = face_recognizer.process_frame(test_image)

    info = face_recognizer.get_known_faces_info()

    print(f"✅ Test completed. Results: {results}")
    print(f"📊 Known faces: {info['names']} (total {info['total_faces']})")
    print(f"🔢 Embedding dimension: {info['embedding_dim']} on device {info['device']} (detector: {info['detector']})")

    return results

if __name__ == "__main__":
    cli_device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    test_face_recognition(cli_device)