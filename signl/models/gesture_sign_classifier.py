"""
Gesture-based Sign Language Classifier using MediaPipe hand landmarks
This classifier uses hand geometry and finger positions to recognize ASL signs
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


class GestureSignClassifier:
    """
    Simple but effective sign language classifier using hand geometry.
    Recognizes common ASL signs based on finger positions and hand shapes.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.sequence_buffer = deque(maxlen=10)  # Buffer for temporal smoothing
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_cooldown = 1.5  # seconds between same predictions
        self.total_predictions = 0
        
        # ASL alphabet and common words we can recognize
        self.signs = [
            'hello', 'thanks', 'yes', 'no', 'please', 
            'sorry', 'help', 'good', 'bad', 'love',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'okay', 'thumbs_up', 'peace', 'stop', 'come'
        ]
        
        logger.info(f"✅ GestureSignClassifier initialized with {len(self.signs)} recognizable signs")
    
    def _calculate_finger_distances(self, hand_landmarks) -> Dict[str, float]:
        """Calculate distances between key finger points"""
        if not hand_landmarks or len(hand_landmarks) < 21:
            return {}
        
        # Key landmarks for fingers
        WRIST = 0
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        
        INDEX_PIP = 6
        MIDDLE_PIP = 10
        RING_PIP = 14
        PINKY_PIP = 18
        
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
        
        # Extract positions
        wrist = [hand_landmarks[WRIST].x, hand_landmarks[WRIST].y, hand_landmarks[WRIST].z]
        thumb_tip = [hand_landmarks[THUMB_TIP].x, hand_landmarks[THUMB_TIP].y, hand_landmarks[THUMB_TIP].z]
        index_tip = [hand_landmarks[INDEX_TIP].x, hand_landmarks[INDEX_TIP].y, hand_landmarks[INDEX_TIP].z]
        middle_tip = [hand_landmarks[MIDDLE_TIP].x, hand_landmarks[MIDDLE_TIP].y, hand_landmarks[MIDDLE_TIP].z]
        ring_tip = [hand_landmarks[RING_TIP].x, hand_landmarks[RING_TIP].y, hand_landmarks[RING_TIP].z]
        pinky_tip = [hand_landmarks[PINKY_TIP].x, hand_landmarks[PINKY_TIP].y, hand_landmarks[PINKY_TIP].z]
        
        index_pip = [hand_landmarks[INDEX_PIP].x, hand_landmarks[INDEX_PIP].y, hand_landmarks[INDEX_PIP].z]
        middle_pip = [hand_landmarks[MIDDLE_PIP].x, hand_landmarks[MIDDLE_PIP].y, hand_landmarks[MIDDLE_PIP].z]
        ring_pip = [hand_landmarks[RING_PIP].x, hand_landmarks[RING_PIP].y, hand_landmarks[RING_PIP].z]
        pinky_pip = [hand_landmarks[PINKY_PIP].x, hand_landmarks[PINKY_PIP].y, hand_landmarks[PINKY_PIP].z]
        
        return {
            'thumb_to_index': distance(thumb_tip, index_tip),
            'thumb_to_middle': distance(thumb_tip, middle_tip),
            'thumb_to_ring': distance(thumb_tip, ring_tip),
            'thumb_to_pinky': distance(thumb_tip, pinky_tip),
            'index_to_wrist': distance(index_tip, wrist),
            'middle_to_wrist': distance(middle_tip, wrist),
            'ring_to_wrist': distance(ring_tip, wrist),
            'pinky_to_wrist': distance(pinky_tip, wrist),
            'index_pip_to_wrist': distance(index_pip, wrist),
            'middle_pip_to_wrist': distance(middle_pip, wrist),
            'ring_pip_to_wrist': distance(ring_pip, wrist),
            'pinky_pip_to_wrist': distance(pinky_pip, wrist),
        }
    
    def _is_finger_extended(self, tip_dist: float, pip_dist: float) -> bool:
        """Check if a finger is extended based on tip and PIP distances from wrist"""
        return tip_dist > pip_dist * 1.2
    
    def _recognize_sign(self, hand_landmarks) -> Tuple[Optional[str], float]:
        """
        Recognize sign based on hand geometry
        Returns (sign_name, confidence)
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return None, 0.0
        
        distances = self._calculate_finger_distances(hand_landmarks)
        if not distances:
            return None, 0.0
        
        # Check finger extension states
        index_extended = self._is_finger_extended(
            distances['index_to_wrist'], distances['index_pip_to_wrist']
        )
        middle_extended = self._is_finger_extended(
            distances['middle_to_wrist'], distances['middle_pip_to_wrist']
        )
        ring_extended = self._is_finger_extended(
            distances['ring_to_wrist'], distances['ring_pip_to_wrist']
        )
        pinky_extended = self._is_finger_extended(
            distances['pinky_to_wrist'], distances['pinky_pip_to_wrist']
        )
        
        # Recognition rules based on finger positions
        
        # THUMBS UP - only thumb extended
        if (not index_extended and not middle_extended and 
            not ring_extended and not pinky_extended):
            return 'thumbs_up', 0.85
        
        # PEACE / VICTORY - index and middle extended
        if (index_extended and middle_extended and 
            not ring_extended and not pinky_extended):
            if distances['thumb_to_index'] < 0.15:
                return 'okay', 0.80  # Thumb touching index/middle
            return 'peace', 0.85
        
        # STOP / HAND - all fingers extended
        if (index_extended and middle_extended and 
            ring_extended and pinky_extended):
            return 'stop', 0.80
        
        # POINT / COME - only index extended
        if (index_extended and not middle_extended and 
            not ring_extended and not pinky_extended):
            return 'come', 0.75
        
        # FIST - no fingers extended (A in ASL)
        if (not index_extended and not middle_extended and 
            not ring_extended and not pinky_extended):
            if distances['thumb_to_index'] < 0.1:
                return 'A', 0.75
        
        # OKAY - thumb and index form circle
        if distances['thumb_to_index'] < 0.08 and middle_extended:
            return 'okay', 0.85
        
        # I LOVE YOU - thumb, index, and pinky extended
        if (index_extended and not middle_extended and 
            not ring_extended and pinky_extended):
            return 'love', 0.85
        
        # YES - fist moving up/down (simplified as closed hand)
        if (not index_extended and not middle_extended and 
            not ring_extended and not pinky_extended):
            return 'yes', 0.60
        
        # HELLO - open hand (all fingers extended)
        if (index_extended and middle_extended and 
            ring_extended and pinky_extended):
            return 'hello', 0.70
        
        # Default: uncertain
        return 'uncertain', 0.3
    
    def process_frame(self, mediapipe_results) -> Dict:
        """
        Process MediaPipe results and predict sign
        """
        start_time = time.time()
        
        try:
            # Check for hand landmarks
            if not mediapipe_results:
                return self._no_detection_result()
            
            # Try right hand first, then left hand
            hand_landmarks = None
            hand_type = None
            
            if hasattr(mediapipe_results, 'right_hand_landmarks') and mediapipe_results.right_hand_landmarks:
                hand_landmarks = mediapipe_results.right_hand_landmarks.landmark
                hand_type = 'right'
            elif hasattr(mediapipe_results, 'left_hand_landmarks') and mediapipe_results.left_hand_landmarks:
                hand_landmarks = mediapipe_results.left_hand_landmarks.landmark
                hand_type = 'left'
            
            if not hand_landmarks:
                return self._no_detection_result()
            
            # Recognize sign
            sign, confidence = self._recognize_sign(hand_landmarks)
            
            if sign and confidence >= self.confidence_threshold:
                # Add to sequence buffer for smoothing
                self.sequence_buffer.append((sign, confidence))
                
                # Get most common sign in buffer
                if len(self.sequence_buffer) >= 3:
                    signs_in_buffer = [s[0] for s in self.sequence_buffer]
                    most_common = max(set(signs_in_buffer), key=signs_in_buffer.count)
                    avg_confidence = np.mean([s[1] for s in self.sequence_buffer if s[0] == most_common])
                    
                    # Check cooldown to avoid rapid repeated predictions
                    current_time = time.time()
                    if (most_common != self.last_prediction or 
                        (current_time - self.last_prediction_time) > self.prediction_cooldown):
                        
                        self.last_prediction = most_common
                        self.last_prediction_time = current_time
                        self.total_predictions += 1
                        
                        prediction_time = (time.time() - start_time) * 1000
                        
                        return {
                            'predicted_sign': most_common,
                            'confidence': float(avg_confidence),
                            'hand_type': hand_type,
                            'status': 'Active prediction',
                            'prediction_time_ms': prediction_time,
                            'buffer_size': len(self.sequence_buffer),
                            'method': 'gesture_recognition'
                        }
            
            return self._no_detection_result()
            
        except Exception as e:
            logger.error(f"Error in gesture recognition: {e}")
            return {
                'predicted_sign': 'Error',
                'confidence': 0.0,
                'status': f'Error: {str(e)}',
                'method': 'gesture_recognition'
            }
    
    def _no_detection_result(self) -> Dict:
        """Return result when no sign detected"""
        return {
            'predicted_sign': 'No sign detected',
            'confidence': 0.0,
            'status': 'Waiting for hand gesture',
            'method': 'gesture_recognition'
        }
    
    def reset_sequence(self):
        """Reset the sequence buffer"""
        self.sequence_buffer.clear()
        self.last_prediction = None
        logger.debug("Gesture sequence reset")
    
    def get_model_info(self) -> Dict:
        """Get classifier information"""
        return {
            'model_loaded': True,
            'actions': self.signs,
            'confidence_threshold': self.confidence_threshold,
            'device': 'CPU',
            'backend': 'Gesture Recognition',
            'method': 'Hand Geometry Analysis',
            'status': 'Ready',
            'total_predictions': self.total_predictions,
            'buffer_size': len(self.sequence_buffer)
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for predictions"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Gesture classifier confidence threshold set to {self.confidence_threshold}")
