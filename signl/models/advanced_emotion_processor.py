# src/majorSignL/models/advanced_emotion_processor.py

import cv2
import numpy as np
import torch
import threading
import queue
import time
from collections import deque
from typing import Dict, Optional, Any, Tuple
import logging
from PIL import Image
import json
import os

logger = logging.getLogger(__name__)

class AdvancedEmotionProcessor:
    """
    Advanced emotion detection with fine-tuning, smoothing, and alignment
    Supports both discrete emotions and valence-arousal continuous values
    """
    
    def __init__(self, 
                 device: str = "cuda", 
                 smoothing_window: int = 7,
                 confidence_threshold: float = 0.4,
                 face_size: int = 224,
                 enable_valence_arousal: bool = True):
        """
        Initialize advanced emotion processor
        
        Args:
            device: CUDA device for emotion detection
            smoothing_window: Number of frames for moving average smoothing
            confidence_threshold: Minimum confidence to accept emotion
            face_size: Size to resize face crops (224x224 standard)
            enable_valence_arousal: Enable continuous valence-arousal mode
        """
        self.device = device
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.face_size = face_size
        self.enable_valence_arousal = enable_valence_arousal
        
        # Threading components
        self.input_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=1)
        self.processing_thread = None
        self.is_running = False
        
        # Smoothing history
        self.emotion_history = deque(maxlen=smoothing_window)
        self.valence_history = deque(maxlen=smoothing_window)
        self.arousal_history = deque(maxlen=smoothing_window)
        
        # Current emotion state
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.raw_probabilities = {}
        self.smoothed_probabilities = {}
        self.valence = 0.0  # -1 (negative) to +1 (positive)
        self.arousal = 0.0  # -1 (calm) to +1 (excited)
        
        # Fine-tuning parameters (can be adjusted)
        self.emotion_thresholds = {
            'happy': 0.5,
            'sad': 0.4,
            'angry': 0.45,
            'surprised': 0.5,
            'fearful': 0.4,
            'disgusted': 0.4,
            'neutral': 0.3
        }
        
        # Emotion to valence-arousal mapping
        self.emotion_va_map = {
            'happy': (0.8, 0.6),      # positive, high arousal
            'sad': (-0.7, -0.4),      # negative, low arousal
            'angry': (-0.6, 0.7),     # negative, high arousal
            'surprised': (0.2, 0.8),  # slightly positive, very high arousal
            'fearful': (-0.5, 0.5),   # negative, medium arousal
            'disgusted': (-0.7, 0.2), # negative, low-medium arousal
            'neutral': (0.0, 0.0)     # neutral valence and arousal
        }
        
        # Performance tracking
        self.total_processed = 0
        self.avg_processing_time = 0
        
        # Model initialization flag
        self.model_initialized = False
        
        logger.info(f"🎭 Advanced Emotion Processor - Smoothing: {smoothing_window}, VA-Mode: {enable_valence_arousal}")
        
    def start(self):
        """Start the advanced emotion processing thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("🎭 Advanced emotion processing thread started")
        
    def stop(self):
        """Stop the advanced emotion processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        logger.info("🎭 Advanced emotion processing stopped")
        
    def set_fine_tune_params(self, emotion_thresholds: Dict[str, float] = None, 
                           confidence_threshold: float = None):
        """Fine-tune emotion detection parameters"""
        if emotion_thresholds:
            self.emotion_thresholds.update(emotion_thresholds)
            logger.info(f"🎛️ Updated emotion thresholds: {emotion_thresholds}")
            
        if confidence_threshold:
            self.confidence_threshold = confidence_threshold
            logger.info(f"🎛️ Updated confidence threshold: {confidence_threshold}")
            
    def process_frame_async(self, frame: np.ndarray, face_landmarks=None):
        """
        Queue frame for asynchronous emotion processing with face alignment
        
        Args:
            frame: Input frame
            face_landmarks: MediaPipe face landmarks for precise alignment
        """
        try:
            frame_data = {
                'frame': frame.copy(),
                'face_landmarks': face_landmarks,
                'timestamp': time.time()
            }
            
            # Replace old frame if queue is full
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.input_queue.put_nowait(frame_data)
            
        except queue.Full:
            pass
            
    def get_emotion_data(self) -> Dict[str, Any]:
        """Get current emotion detection results with smoothing and VA values"""
        # Try to get latest results
        try:
            while True:
                try:
                    result = self.result_queue.get_nowait()
                    self.current_emotion = result['emotion']
                    self.emotion_confidence = result['confidence']
                    self.raw_probabilities = result['raw_probabilities']
                    self.smoothed_probabilities = result['smoothed_probabilities']
                    if self.enable_valence_arousal:
                        self.valence = result['valence']
                        self.arousal = result['arousal']
                except queue.Empty:
                    break
        except:
            pass
            
        return {
            'emotion': self.current_emotion,
            'confidence': round(self.emotion_confidence, 3),
            'raw_probabilities': {k: round(v, 3) for k, v in self.raw_probabilities.items()},
            'smoothed_probabilities': {k: round(v, 3) for k, v in self.smoothed_probabilities.items()},
            'valence': round(self.valence, 3) if self.enable_valence_arousal else None,
            'arousal': round(self.arousal, 3) if self.enable_valence_arousal else None,
            'model_loaded': self.model_initialized,
            'total_processed': self.total_processed,
            'avg_processing_time_ms': round(self.avg_processing_time * 1000, 1),
            'fine_tune_params': {
                'emotion_thresholds': self.emotion_thresholds,
                'confidence_threshold': self.confidence_threshold,
                'smoothing_window': self.smoothing_window
            }
        }
    
    def _extract_aligned_face(self, frame: np.ndarray, face_landmarks) -> Optional[np.ndarray]:
        """
        Extract and align face using MediaPipe landmarks
        Force face alignment → crop → resize to 224x224 RGB
        """
        if not face_landmarks:
            return None
            
        try:
            h, w = frame.shape[:2]
            landmarks = face_landmarks.landmark
            
            # Get face bounding box with improved padding
            x_coords = [int(landmark.x * w) for landmark in landmarks]
            y_coords = [int(landmark.y * h) for landmark in landmarks]
            
            # Calculate face bounds with better margins
            face_width = max(x_coords) - min(x_coords)
            face_height = max(y_coords) - min(y_coords)
            
            # Add padding (25% of face size)
            padding_x = int(face_width * 0.25)
            padding_y = int(face_height * 0.3)  # More padding on top for forehead
            
            x1 = max(0, min(x_coords) - padding_x)
            y1 = max(0, min(y_coords) - padding_y)
            x2 = min(w, max(x_coords) + padding_x)
            y2 = min(h, max(y_coords) + int(padding_y * 0.8))  # Less padding on bottom
            
            # Crop face region
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
                
            # Resize to standard size (224x224) and convert to RGB
            face_resized = cv2.resize(face_crop, (self.face_size, self.face_size))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            return face_rgb
            
        except Exception as e:
            logger.error(f"Face alignment error: {e}")
            return None
    
    def _geometric_emotion_detection(self, face_landmarks) -> Dict[str, float]:
        """
        Enhanced geometric emotion detection as fallback
        """
        if not face_landmarks:
            return {'neutral': 1.0}
            
        landmarks = face_landmarks.landmark
        
        # Calculate facial feature ratios
        mouth_ratio = self._calculate_mouth_curvature(landmarks)
        eye_openness = self._calculate_eye_openness(landmarks)
        eyebrow_position = self._calculate_eyebrow_position(landmarks)
        jaw_tension = self._calculate_jaw_tension(landmarks)
        
        # Initialize probabilities
        probs = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'fearful': 0.0,
            'disgusted': 0.0,
            'neutral': 0.1
        }
        
        # Enhanced emotion rules
        if mouth_ratio > 0.02 and eye_openness > 0.15:  # Smiling with normal eyes
            probs['happy'] = min(0.9, mouth_ratio * 30 + 0.3)
            
        elif mouth_ratio < -0.015:  # Mouth corners down
            if eyebrow_position < -0.01:  # Eyebrows down too
                probs['sad'] = min(0.8, abs(mouth_ratio) * 25 + abs(eyebrow_position) * 20)
            else:
                probs['disgusted'] = min(0.7, abs(mouth_ratio) * 20)
                
        elif eyebrow_position < -0.015 and jaw_tension > 0.02:  # Angry expression
            probs['angry'] = min(0.8, abs(eyebrow_position) * 30 + jaw_tension * 15)
            
        elif eye_openness > 0.25 and eyebrow_position > 0.02:  # Wide eyes, raised brows
            probs['surprised'] = min(0.8, eye_openness * 20 + eyebrow_position * 25)
            
        elif eye_openness > 0.2 and mouth_ratio > 0.01 and eyebrow_position > 0.005:  # Wide eyes, slight smile
            probs['fearful'] = min(0.7, eye_openness * 15 + eyebrow_position * 20)
        
        # Ensure neutral has reasonable probability if others are low
        if max(probs.values()) < 0.4:
            probs['neutral'] = 0.7
            
        return probs
    
    def _calculate_mouth_curvature(self, landmarks):
        """Calculate mouth curvature for smile/frown detection"""
        try:
            # Mouth corners and center points
            left_corner = landmarks[61]   # Left mouth corner
            right_corner = landmarks[291] # Right mouth corner
            top_lip = landmarks[13]       # Top lip center
            bottom_lip = landmarks[14]    # Bottom lip center
            
            # Calculate curvature
            mouth_center_y = (top_lip.y + bottom_lip.y) / 2
            corners_y = (left_corner.y + right_corner.y) / 2
            mouth_width = abs(right_corner.x - left_corner.x)
            
            return (mouth_center_y - corners_y) / mouth_width if mouth_width > 0 else 0
        except:
            return 0
    
    def _calculate_eye_openness(self, landmarks):
        """Calculate average eye openness"""
        try:
            # Left eye
            left_top = landmarks[159].y
            left_bottom = landmarks[145].y
            left_width = abs(landmarks[33].x - landmarks[133].x)
            
            # Right eye  
            right_top = landmarks[386].y
            right_bottom = landmarks[374].y
            right_width = abs(landmarks[362].x - landmarks[263].x)
            
            # Calculate ratios
            left_ratio = abs(left_top - left_bottom) / left_width if left_width > 0 else 0
            right_ratio = abs(right_top - right_bottom) / right_width if right_width > 0 else 0
            
            return (left_ratio + right_ratio) / 2
        except:
            return 0.15  # Default normal eye openness
    
    def _calculate_eyebrow_position(self, landmarks):
        """Calculate eyebrow position relative to eyes"""
        try:
            # Left side
            left_brow = landmarks[70].y
            left_eye = landmarks[159].y
            
            # Right side
            right_brow = landmarks[300].y  
            right_eye = landmarks[386].y
            
            # Calculate distances
            left_dist = left_eye - left_brow
            right_dist = right_eye - right_brow
            
            return (left_dist + right_dist) / 2
        except:
            return 0
    
    def _calculate_jaw_tension(self, landmarks):
        """Calculate jaw tension for anger detection"""
        try:
            # Jaw points
            left_jaw = landmarks[172]
            right_jaw = landmarks[397]
            chin = landmarks[18]
            
            # Calculate jaw width relative to face height
            jaw_width = abs(right_jaw.x - left_jaw.x)
            face_height = abs(landmarks[10].y - landmarks[152].y)  # Forehead to chin
            
            return jaw_width / face_height if face_height > 0 else 0
        except:
            return 0
    
    def _processing_loop(self):
        """Main processing loop with smoothing and alignment"""
        logger.info("🎭 Starting advanced emotion processing loop...")
        
        self.model_initialized = True  # Using geometric detection
        processing_times = []
        
        while self.is_running:
            try:
                frame_data = self.input_queue.get(timeout=0.1)
                start_time = time.time()
                
                frame = frame_data['frame']
                face_landmarks = frame_data.get('face_landmarks')
                
                # Extract aligned face
                aligned_face = self._extract_aligned_face(frame, face_landmarks)
                
                if aligned_face is None:
                    continue
                
                # Get raw emotion probabilities using geometric method
                raw_probs = self._geometric_emotion_detection(face_landmarks)
                
                # Add to history for smoothing
                self.emotion_history.append(raw_probs)
                
                # Calculate smoothed probabilities
                if len(self.emotion_history) > 0:
                    # Average probabilities across history
                    smoothed_probs = {}
                    for emotion in raw_probs.keys():
                        values = [frame_probs.get(emotion, 0) for frame_probs in self.emotion_history]
                        smoothed_probs[emotion] = np.mean(values)
                else:
                    smoothed_probs = raw_probs.copy()
                
                # Apply fine-tuning thresholds
                final_emotion = 'neutral'
                final_confidence = smoothed_probs.get('neutral', 0.5)
                
                for emotion, prob in smoothed_probs.items():
                    threshold = self.emotion_thresholds.get(emotion, 0.5)
                    if prob > threshold and prob > final_confidence:
                        final_emotion = emotion
                        final_confidence = prob
                
                # Calculate valence-arousal if enabled
                valence, arousal = 0.0, 0.0
                if self.enable_valence_arousal:
                    # Weighted average based on emotion probabilities
                    total_weight = sum(smoothed_probs.values())
                    if total_weight > 0:
                        valence = sum(prob * self.emotion_va_map[emotion][0] 
                                    for emotion, prob in smoothed_probs.items()) / total_weight
                        arousal = sum(prob * self.emotion_va_map[emotion][1] 
                                    for emotion, prob in smoothed_probs.items()) / total_weight
                    
                    # Add to history and smooth
                    self.valence_history.append(valence)
                    self.arousal_history.append(arousal)
                    
                    valence = np.mean(self.valence_history)
                    arousal = np.mean(self.arousal_history)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                if len(processing_times) > 30:
                    processing_times = processing_times[-30:]
                    
                self.avg_processing_time = np.mean(processing_times)
                self.total_processed += 1
                
                # Create result
                result = {
                    'emotion': final_emotion,
                    'confidence': final_confidence,
                    'raw_probabilities': raw_probs,
                    'smoothed_probabilities': smoothed_probs,
                    'valence': valence,
                    'arousal': arousal,
                    'timestamp': time.time()
                }
                
                # Update result queue
                try:
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Advanced emotion processing error: {e}")
                continue
                
        logger.info("🎭 Advanced emotion processing loop stopped")
        
    def save_fine_tune_params(self, filepath: str):
        """Save current fine-tuning parameters to file"""
        params = {
            'emotion_thresholds': self.emotion_thresholds,
            'confidence_threshold': self.confidence_threshold,
            'smoothing_window': self.smoothing_window
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        logger.info(f"💾 Fine-tune parameters saved to {filepath}")
        
    def load_fine_tune_params(self, filepath: str):
        """Load fine-tuning parameters from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                params = json.load(f)
            
            self.emotion_thresholds.update(params.get('emotion_thresholds', {}))
            self.confidence_threshold = params.get('confidence_threshold', self.confidence_threshold)
            
            logger.info(f"📁 Fine-tune parameters loaded from {filepath}")
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop()
