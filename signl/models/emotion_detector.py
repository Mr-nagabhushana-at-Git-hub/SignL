"""
Emotion Detection Module
Extracts emotions from MediaPipe face landmarks
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class EmotionDetector:
    """
    Real-time emotion detection using MediaPipe face landmarks
    Analyzes facial geometry to classify emotions
    """
    
    def __init__(self):
        self.emotions = {
            'neutral': 0.0,
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'disgusted': 0.0,
            'fearful': 0.0
        }
        
        # Fine-tuning thresholds for different emotions
        self.emotion_thresholds = {
            'happy': 0.3,
            'sad': 0.25,
            'angry': 0.35,
            'surprised': 0.4,
            'disgusted': 0.3,
            'fearful': 0.3,
            'neutral': 0.2
        }
        
        self.confidence_threshold = 0.5
        
    def update_thresholds(self, emotion_thresholds: dict, confidence_threshold: float = None):
        """Update emotion detection thresholds for fine-tuning"""
        if emotion_thresholds:
            self.emotion_thresholds.update(emotion_thresholds)
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
    def analyze_emotion(self, face_landmarks) -> Dict[str, float]:
        """
        Enhanced facial landmark analysis for better emotion detection
        Evaluates ALL emotions independently and returns the strongest
        """
        if not face_landmarks:
            return self.emotions.copy()
            
        # Extract key facial points for emotion analysis
        landmarks = face_landmarks.landmark
        
        # Calculate enhanced facial ratios and features
        mouth_ratio = self._calculate_mouth_ratio(landmarks)
        eye_ratio = self._calculate_eye_ratio(landmarks)
        eyebrow_ratio = self._calculate_eyebrow_ratio(landmarks)
        cheek_ratio = self._calculate_cheek_ratio(landmarks)
        
        # Reset emotions
        emotions = self.emotions.copy()
        
        # Evaluate ALL emotions independently
        if mouth_ratio > 0.012 and cheek_ratio > 0.008:
            emotions['happy'] = min(1.0, mouth_ratio * 40 + cheek_ratio * 30)
            
        if mouth_ratio < -0.006 and eyebrow_ratio < -0.002:
            emotions['sad'] = min(1.0, abs(mouth_ratio) * 45 + abs(eyebrow_ratio) * 35)
            
        if eye_ratio > 0.016 and eyebrow_ratio > 0.008:
            emotions['surprised'] = min(1.0, eye_ratio * 50 + eyebrow_ratio * 40)
            
        if eyebrow_ratio < -0.008:
            emotions['angry'] = min(1.0, abs(eyebrow_ratio) * 50)
            if abs(mouth_ratio) < 0.008:
                emotions['angry'] = min(1.0, emotions['angry'] + 0.2)
            
        if eye_ratio > 0.013 and eyebrow_ratio > 0.003 and abs(mouth_ratio) < 0.012:
            emotions['fearful'] = min(1.0, eye_ratio * 35 + eyebrow_ratio * 25)
        
        if mouth_ratio < -0.003 and cheek_ratio < -0.005:
            emotions['disgusted'] = min(1.0, abs(mouth_ratio) * 30 + abs(cheek_ratio) * 25)
            
        emotions['neutral'] = 0.3
        
        max_emotion_value = max(emotions.values())
        if max_emotion_value < 0.25:
            emotions['neutral'] = 0.7
            
        return emotions
    
    def _calculate_mouth_ratio(self, landmarks):
        """Calculate mouth curvature (smile/frown detection)"""
        try:
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            top_lip = landmarks[13]
            bottom_lip = landmarks[14]
            
            mouth_width = abs(right_corner.x - left_corner.x)
            lip_center_y = (top_lip.y + bottom_lip.y) / 2
            corner_y = (left_corner.y + right_corner.y) / 2
            
            return (lip_center_y - corner_y) / mouth_width if mouth_width > 0 else 0
        except (IndexError, AttributeError):
            return 0
    
    def _calculate_eye_ratio(self, landmarks):
        """Calculate eye openness ratio"""
        try:
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            left_eye_left = landmarks[133]
            left_eye_right = landmarks[33]
            
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            right_eye_left = landmarks[362]
            right_eye_right = landmarks[263]
            
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            left_eye_width = abs(left_eye_right.x - left_eye_left.x)
            
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            right_eye_width = abs(right_eye_right.x - right_eye_left.x)
            
            left_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            right_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            
            return (left_ratio + right_ratio) / 2
        except (IndexError, AttributeError):
            return 0
    
    def _calculate_eyebrow_ratio(self, landmarks):
        """Calculate eyebrow position (raised/lowered)"""
        try:
            left_eyebrow = landmarks[70]
            left_eye_top = landmarks[159]
            right_eyebrow = landmarks[300]
            right_eye_top = landmarks[386]
            
            left_distance = left_eye_top.y - left_eyebrow.y
            right_distance = right_eye_top.y - right_eyebrow.y
            
            return (left_distance + right_distance) / 2
        except (IndexError, AttributeError):
            return 0
    
    def _calculate_cheek_ratio(self, landmarks):
        """Calculate cheek elevation (for smile detection)"""
        try:
            left_cheek = landmarks[116]
            right_cheek = landmarks[345]
            nose_tip = landmarks[1]
            
            cheek_y = (left_cheek.y + right_cheek.y) / 2
            elevation = nose_tip.y - cheek_y
            
            return elevation
        except (IndexError, AttributeError):
            return 0
