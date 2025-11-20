# Real-time Sign Language Recognition using MediaPipe Hands
# Supports ASL (American Sign Language) gestures

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, Optional, List, Tuple
from collections import deque
import time

logger = logging.getLogger(__name__)

class SignLanguageRecognizer:
    """
    Real-time Sign Language Recognition using MediaPipe Hands
    Recognizes common ASL gestures based on hand landmarks and positions
    """
    
    def __init__(self):
        """Initialize MediaPipe Hands and sign recognition system"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector with optimized settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Gesture buffer for temporal smoothing
        self.gesture_buffer = deque(maxlen=10)
        self.current_gesture = "No hands detected"
        self.confidence = 0.0
        
        # Translation history
        self.translation_history = deque(maxlen=50)
        self.last_translation_time = 0
        self.translation_cooldown = 1.5  # seconds between translations
        
        # Performance tracking
        self.frame_count = 0
        self.total_process_time = 0
        
        logger.info("✅ Sign Language Recognizer initialized with MediaPipe Hands")
    
    def recognize_gesture(self, hand_landmarks, handedness: str) -> Tuple[str, float]:
        """
        Recognize ASL gesture from hand landmarks
        Returns (gesture_name, confidence)
        """
        if not hand_landmarks:
            return ("No hands detected", 0.0)
        
        # Extract key landmark positions
        landmarks = hand_landmarks.landmark
        
        # Finger tip and base indices
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        
        THUMB_IP = 3
        INDEX_MCP = 5
        MIDDLE_MCP = 9
        RING_MCP = 13
        PINKY_MCP = 17
        WRIST = 0
        
        # Helper functions
        def is_finger_extended(tip_idx, mcp_idx):
            """Check if finger is extended based on tip vs MCP position"""
            return landmarks[tip_idx].y < landmarks[mcp_idx].y
        
        def is_thumb_extended():
            """Special check for thumb"""
            if handedness == "Right":
                return landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
            else:
                return landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x
        
        def fingers_together(tip1_idx, tip2_idx, threshold=0.05):
            """Check if two fingertips are close together"""
            dx = landmarks[tip1_idx].x - landmarks[tip2_idx].x
            dy = landmarks[tip1_idx].y - landmarks[tip2_idx].y
            return (dx*dx + dy*dy) < threshold*threshold
        
        def fingers_apart(tip1_idx, tip2_idx, threshold=0.1):
            """Check if two fingertips are far apart"""
            dx = landmarks[tip1_idx].x - landmarks[tip2_idx].x
            dy = landmarks[tip1_idx].y - landmarks[tip2_idx].y
            return (dx*dx + dy*dy) > threshold*threshold
        
        # Count extended fingers
        thumb_extended = is_thumb_extended()
        index_extended = is_finger_extended(INDEX_TIP, INDEX_MCP)
        middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_MCP)
        ring_extended = is_finger_extended(RING_TIP, RING_MCP)
        pinky_extended = is_finger_extended(PINKY_TIP, PINKY_MCP)
        
        extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
        
        # Gesture Recognition Logic
        
        # THUMBS UP (only thumb extended, others closed)
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return ("THUMBS UP / Good / Yes", 0.9)
        
        # THUMBS DOWN (thumb down, others closed)
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            thumb_down = landmarks[THUMB_TIP].y > landmarks[WRIST].y
            if thumb_down:
                return ("THUMBS DOWN / Bad / No", 0.9)
        
        # PEACE / VICTORY (index and middle extended, others closed)
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            if fingers_apart(INDEX_TIP, MIDDLE_TIP, 0.08):
                return ("PEACE / Victory / 2", 0.85)
        
        # OKAY (thumb and index form circle, others extended)
        if fingers_together(THUMB_TIP, INDEX_TIP, 0.04) and middle_extended and ring_extended and pinky_extended:
            return ("OKAY / Perfect", 0.85)
        
        # POINTING (only index extended)
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return ("POINTING / You / There", 0.8)
        
        # FIST / CLOSED HAND (no fingers extended)
        if extended_count == 0:
            return ("FIST / Stop / Letter S", 0.85)
        
        # OPEN PALM (all fingers extended)
        if extended_count == 5:
            # Check if fingers are spread
            if fingers_apart(INDEX_TIP, MIDDLE_TIP, 0.06) and fingers_apart(MIDDLE_TIP, RING_TIP, 0.06):
                return ("OPEN HAND / Hello / Stop / 5", 0.85)
            else:
                return ("FLAT HAND / Letter B", 0.8)
        
        # I LOVE YOU (thumb, index, pinky extended)
        if thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
            return ("I LOVE YOU", 0.9)
        
        # ROCK/HORNS (index and pinky extended, others closed)
        if not thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
            return ("ROCK / Horns / Letter Y", 0.85)
        
        # THREE (thumb, index, middle extended)
        if thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            return ("THREE / Letter W", 0.8)
        
        # FOUR (all fingers except thumb extended)
        if not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return ("FOUR / Letter 4", 0.8)
        
        # CALL ME (thumb and pinky extended, others closed)
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return ("CALL ME / Letter Y", 0.8)
        
        # Default - unknown gesture
        return (f"Unknown gesture ({extended_count} fingers)", 0.5)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return sign language recognition results
        
        Args:
            frame: BGR image from camera
            
        Returns:
            dict with keys: gesture, confidence, text, hands_detected, processing_time_ms
        """
        start_time = time.time()
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(frame_rgb)
        
        gesture = "No hands detected"
        confidence = 0.0
        text_output = ""
        hands_detected = 0
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hands_detected = len(results.multi_hand_landmarks)
            
            # Process each hand (prioritize right hand for single-hand gestures)
            best_gesture = None
            best_confidence = 0.0
            
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness_info.classification[0].label  # "Left" or "Right"
                
                detected_gesture, gest_confidence = self.recognize_gesture(hand_landmarks, hand_label)
                
                if gest_confidence > best_confidence:
                    best_gesture = detected_gesture
                    best_confidence = gest_confidence
            
            if best_gesture:
                gesture = best_gesture
                confidence = best_confidence
                
                # Add to buffer for smoothing
                self.gesture_buffer.append((gesture, confidence))
                
                # Get most common gesture from buffer
                if len(self.gesture_buffer) >= 5:
                    gesture_votes = {}
                    for g, c in self.gesture_buffer:
                        if g not in gesture_votes:
                            gesture_votes[g] = []
                        gesture_votes[g].append(c)
                    
                    # Find gesture with highest average confidence
                    best_avg = 0
                    for g, confs in gesture_votes.items():
                        avg_conf = sum(confs) / len(confs)
                        if avg_conf > best_avg:
                            best_avg = avg_conf
                            gesture = g
                            confidence = avg_conf
                
                # Add to translation history if stable and enough time passed
                current_time = time.time()
                if confidence > 0.7 and (current_time - self.last_translation_time) > self.translation_cooldown:
                    if gesture != self.current_gesture:
                        self.current_gesture = gesture
                        self.translation_history.append({
                            'gesture': gesture,
                            'confidence': confidence,
                            'timestamp': current_time
                        })
                        self.last_translation_time = current_time
                        text_output = gesture.split('/')[0].strip()  # Extract main meaning
        
        # Performance tracking
        processing_time = (time.time() - start_time) * 1000
        self.frame_count += 1
        self.total_process_time += processing_time
        
        return {
            'gesture': gesture,
            'confidence': confidence,
            'text': text_output,
            'hands_detected': hands_detected,
            'processing_time_ms': round(processing_time, 2),
            'frame_count': self.frame_count,
            'translation_history': list(self.translation_history)
        }
    
    def draw_hands(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks on frame"""
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
    
    def get_translation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent translation history"""
        return list(self.translation_history)[-limit:]
    
    def clear_history(self):
        """Clear translation history"""
        self.translation_history.clear()
        self.gesture_buffer.clear()
        self.current_gesture = "No hands detected"
        logger.info("Translation history cleared")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time = self.total_process_time / max(self.frame_count, 1)
        return {
            'frames_processed': self.frame_count,
            'avg_processing_time_ms': round(avg_time, 2),
            'translations_made': len(self.translation_history)
        }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'hands'):
            self.hands.close()
