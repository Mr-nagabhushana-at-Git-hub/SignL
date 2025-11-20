# src/majorSignL/utils/mediapipe_processor.py

import cv2
import mediapipe as mp
import numpy as np
from majorSignL.utils.one_euro_filter import OneEuroFilter
from majorSignL.models.advanced_emotion_processor import AdvancedEmotionProcessor
import threading
from typing import Optional, Dict, Any
import logging
import math

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
        
        # Debug: Log ratios occasionally
        # Uncomment for detailed debugging:
        # logger.debug(f"Ratios - mouth:{mouth_ratio:.3f} eye:{eye_ratio:.3f} brow:{eyebrow_ratio:.3f} cheek:{cheek_ratio:.3f}")
        
        # Reset emotions
        emotions = self.emotions.copy()
        
        # Evaluate ALL emotions independently (not elif chain)
        # Happy: Mouth corners up, cheeks raised
        if mouth_ratio > 0.012 and cheek_ratio > 0.008:
            emotions['happy'] = min(1.0, mouth_ratio * 40 + cheek_ratio * 30)
            
        # Sad: Mouth corners down, eyebrows down
        if mouth_ratio < -0.006 and eyebrow_ratio < -0.002:
            emotions['sad'] = min(1.0, abs(mouth_ratio) * 45 + abs(eyebrow_ratio) * 35)
            
        # Surprised: Eyes wide, eyebrows up
        if eye_ratio > 0.016 and eyebrow_ratio > 0.008:
            emotions['surprised'] = min(1.0, eye_ratio * 50 + eyebrow_ratio * 40)
            
        # Angry: Eyebrows down, mouth tight or frown
        if eyebrow_ratio < -0.008:
            emotions['angry'] = min(1.0, abs(eyebrow_ratio) * 50)
            # Bonus if mouth is tight or frowning
            if abs(mouth_ratio) < 0.008:
                emotions['angry'] = min(1.0, emotions['angry'] + 0.2)
            
        # Fearful: Eyes wide, mouth slightly open, eyebrows slightly raised
        if eye_ratio > 0.013 and eyebrow_ratio > 0.003 and abs(mouth_ratio) < 0.012:
            emotions['fearful'] = min(1.0, eye_ratio * 35 + eyebrow_ratio * 25)
        
        # Disgusted: Nose wrinkle (approximated), mouth asymmetric
        if mouth_ratio < -0.003 and cheek_ratio < -0.005:
            emotions['disgusted'] = min(1.0, abs(mouth_ratio) * 30 + abs(cheek_ratio) * 25)
            
        # Neutral: baseline when no strong features detected
        emotions['neutral'] = 0.3
        
        # If no emotion is strong, boost neutral
        max_emotion_value = max(emotions.values())
        if max_emotion_value < 0.25:
            emotions['neutral'] = 0.7
            
        return emotions
    
    def _calculate_mouth_ratio(self, landmarks):
        """Calculate mouth curvature (smile/frown detection)"""
        try:
            # Mouth corners and center
            left_corner = landmarks[61]   # Left mouth corner
            right_corner = landmarks[291] # Right mouth corner
            top_lip = landmarks[13]       # Top lip center
            bottom_lip = landmarks[14]    # Bottom lip center
            
            # Calculate mouth width and height
            mouth_width = abs(right_corner.x - left_corner.x)
            mouth_height = abs(top_lip.y - bottom_lip.y)
            
            # Calculate corner elevation relative to center
            lip_center_y = (top_lip.y + bottom_lip.y) / 2
            corner_y = (left_corner.y + right_corner.y) / 2
            
            return (lip_center_y - corner_y) / mouth_width if mouth_width > 0 else 0
        except (IndexError, AttributeError) as e:
            logger.debug(f"Mouth ratio calculation error: {e}")
            return 0
    
    def _calculate_eye_ratio(self, landmarks):
        """Calculate eye openness ratio"""
        try:
            # Left eye points
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            left_eye_left = landmarks[133]
            left_eye_right = landmarks[33]
            
            # Right eye points
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            right_eye_left = landmarks[362]
            right_eye_right = landmarks[263]
            
            # Calculate eye aspect ratios
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            left_eye_width = abs(left_eye_right.x - left_eye_left.x)
            
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            right_eye_width = abs(right_eye_right.x - right_eye_left.x)
            
            left_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            right_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            
            return (left_ratio + right_ratio) / 2
        except (IndexError, AttributeError) as e:
            logger.debug(f"Eye ratio calculation error: {e}")
            return 0
    
    def _calculate_eyebrow_ratio(self, landmarks):
        """Calculate eyebrow position (raised/lowered)"""
        try:
            # Left eyebrow points
            left_eyebrow = landmarks[70]
            left_eye_top = landmarks[159]
            
            # Right eyebrow points  
            right_eyebrow = landmarks[300]
            right_eye_top = landmarks[386]
            
            # Calculate eyebrow distances from eyes
            left_distance = left_eye_top.y - left_eyebrow.y
            right_distance = right_eye_top.y - right_eyebrow.y
            
            return (left_distance + right_distance) / 2
        except (IndexError, AttributeError) as e:
            logger.debug(f"Eyebrow ratio calculation error: {e}")
            return 0
    
    def _calculate_cheek_ratio(self, landmarks):
        """Calculate cheek elevation (for smile detection)"""
        try:
            # Cheek points
            left_cheek = landmarks[116]
            right_cheek = landmarks[345]
            nose_tip = landmarks[1]
            
            # Calculate cheek elevation relative to nose
            cheek_y = (left_cheek.y + right_cheek.y) / 2
            elevation = nose_tip.y - cheek_y
            
            return elevation
        except (IndexError, AttributeError) as e:
            logger.debug(f"Cheek ratio calculation error: {e}")
            return 0

class MediaPipeProcessor:
    def __init__(self, enable_gpu: bool = True, use_mesh: bool = True):
        """
        Enhanced MediaPipe processor with GPU acceleration and mesh visualization
        
        Args:
            enable_gpu: Use GPU acceleration if available
            use_mesh: Enable detailed hand mesh visualization for better tracking
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # GPU-optimized MediaPipe for dynamic face mesh
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,              # Higher complexity for better face mesh
            smooth_landmarks=True,           # Built-in smoothing 
            enable_segmentation=False,       
            smooth_segmentation=False,
            refine_face_landmarks=True,      # Enable for detailed face mesh
            min_detection_confidence=0.5,    
            min_tracking_confidence=0.7      # Higher tracking for stable mesh
        )
        
        # Dedicated hands processor - DISABLED for performance
        self.hands = None  # Disabled to eliminate extra processing
        
        # Lightweight emotion detection (no separate thread to avoid lag)
        self.geometric_emotion = EmotionDetector()
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        # Emotion smoothing and hysteresis - reduced for faster response
        self._emotion_ema = {k: 0.0 for k in ['happy','sad','angry','surprised','fearful','disgusted','neutral']}
        self._emotion_alpha = 0.5  # EMA weight - increased for faster response
        self._emotion_last_switch_ts = 0.0
        self._emotion_min_hold_s = 0.5  # Reduced from 1.0 to 0.5 for faster switching
        
        # Simple smoothing without heavy processing
        self.emotion_history = []
        self.max_history = 3  # Reduced from 5 to 3 for faster response
        
        # Performance optimization - GPU rendering support
        self.use_mesh = use_mesh
        self.enable_gpu = enable_gpu
        self.landmark_filters = {}
        self.filter_freq = 30
        self.latest_results = None
        self.frame_count = 0
        
        # Dynamic mesh colors (changes based on movement)
        self.base_face_color = (80, 256, 121)
        self.face_intensity = 0.5
        
        # Hand mesh connections for detailed visualization
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        
        logger.info(f"🔥 MediaPipe initialized - GPU: {enable_gpu}, Dynamic Face Mesh: ENABLED")

    def _get_dynamic_face_colors(self, face_landmarks):
        """Calculate dynamic colors based on face movement intensity"""
        try:
            if face_landmarks and len(face_landmarks.landmark) > 10:
                # Calculate movement intensity based on nose tip (landmark 1)
                nose_tip = face_landmarks.landmark[1]
                movement_factor = abs(nose_tip.x - 0.5) + abs(nose_tip.y - 0.5)
                
                # Dynamic color intensity (0.3 to 1.0)
                intensity = min(0.3 + movement_factor * 2, 1.0)
                
                # Dynamic colors: Green to Cyan based on movement
                mesh_color = (int(80 * intensity), int(256 * intensity), int(121 + 134 * movement_factor))
                contour_color = (int(80 * intensity), int(256 * intensity), int(121 * intensity))
                
                return mesh_color, contour_color
            else:
                return (80, 256, 121), (80, 256, 121)
        except:
            return (80, 256, 121), (80, 256, 121)

    def _apply_filter(self, landmarks_proto, landmark_type):
        """Apply OneEuro filtering with stable settings for smoother tracking"""
        if landmarks_proto is None:
            return
        
        for i, landmark in enumerate(landmarks_proto.landmark):
            key = f"{landmark_type}_{i}"
            if key not in self.landmark_filters:
                # More stable filter settings - correct parameter names
                self.landmark_filters[key] = OneEuroFilter(
                    freq=self.filter_freq, 
                    min_cutoff=0.5,     # Correct parameter name: min_cutoff
                    beta=0.1,           # Lower beta for less responsiveness
                    d_cutoff=1.0        # Correct parameter name: d_cutoff
                )
            
            raw_coords = np.array([landmark.x, landmark.y, landmark.z])
            filtered_coords = self.landmark_filters[key](raw_coords)
            landmark.x, landmark.y, landmark.z = filtered_coords
    
    def _draw_fingertip_highlights(self, image, hand_landmarks, color):
        """Draw enhanced fingertip visualization for better hand tracking"""
        try:
            h, w, _ = image.shape
            # MediaPipe hand landmark indices for fingertips
            fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            
            for tip_idx in fingertip_indices:
                if tip_idx < len(hand_landmarks.landmark):
                    landmark = hand_landmarks.landmark[tip_idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    # Draw larger highlight circles for fingertips
                    cv2.circle(image, (x, y), 8, color, -1)  # Filled circle
                    cv2.circle(image, (x, y), 10, (255, 255, 255), 2)  # White border
                    
        except Exception as e:
            logger.error(f"Fingertip highlighting error: {e}")
    
    def _draw_palm_mesh_structure(self, image, hand_landmarks, base_color):
        """Draw enhanced palm mesh structure for detailed hand visualization"""
        try:
            h, w, _ = image.shape
            
            # Define palm mesh connections (creating internal palm structure)
            palm_mesh_connections = [
                # Wrist to base connections
                (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
                # Cross-palm connections
                (1, 5), (5, 9), (9, 13), (13, 17),
                # Inner palm mesh
                (2, 5), (5, 6), (6, 9), (9, 10), (10, 13), (13, 14), (14, 17),
                # Additional mesh details
                (1, 2), (2, 3), (5, 6), (6, 7), (9, 10), (10, 11), (13, 14), (14, 15), (17, 18)
            ]
            
            # Draw palm mesh lines with varying thickness for depth
            for start_idx, end_idx in palm_mesh_connections:
                if start_idx < len(hand_landmarks.landmark) and end_idx < len(hand_landmarks.landmark):
                    start_point = hand_landmarks.landmark[start_idx]
                    end_point = hand_landmarks.landmark[end_idx]
                    
                    start_x = int(start_point.x * w)
                    start_y = int(start_point.y * h)
                    end_x = int(end_point.x * w)
                    end_y = int(end_point.y * h)
                    
                    # Varying line thickness based on connection type
                    thickness = 2 if start_idx == 0 or end_idx == 0 else 1
                    cv2.line(image, (start_x, start_y), (end_x, end_y), base_color, thickness)
                    
        except Exception as e:
            logger.error(f"Palm mesh error: {e}")
    
    def _draw_emotion_overlay(self, image, face_landmarks):
        """
        Draw emotion detection overlay on the face
        """
        try:
            if not face_landmarks or self.emotion_confidence < 0.3:
                return
                
            # Get face bounding box for text positioning
            landmarks = face_landmarks.landmark
            h, w, _ = image.shape
            
            # Find face bounds
            x_coords = [int(landmark.x * w) for landmark in landmarks]
            y_coords = [int(landmark.y * h) for landmark in landmarks]
            
            face_left = min(x_coords)
            face_top = min(y_coords)
            face_width = max(x_coords) - face_left
            
            # Emotion colors (removed emojis - OpenCV can't render Unicode)
            emotion_config = {
                'happy': {'color': (0, 255, 0), 'symbol': ':)'},
                'sad': {'color': (255, 100, 100), 'symbol': ':('},
                'angry': {'color': (0, 0, 255), 'symbol': '>:('},
                'surprised': {'color': (0, 255, 255), 'symbol': ':O'},
                'disgusted': {'color': (128, 0, 128), 'symbol': ':P'},
                'fearful': {'color': (255, 165, 0), 'symbol': 'D:'},
                'neutral': {'color': (200, 200, 200), 'symbol': ':|'}
            }
            
            config = emotion_config.get(self.current_emotion, emotion_config['neutral'])
            
            # Draw emotion label above face
            text = f"{config['symbol']} {self.current_emotion.upper()}"
            confidence_text = f"{self.emotion_confidence:.0%}"
            
            # Position text above face
            text_x = face_left + face_width // 4
            text_y = face_top - 40
            
            # Draw background rectangle
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(image, 
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         (0, 0, 0), -1)
            
            # Draw emotion text
            cv2.putText(image, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, config['color'], 2)
            
            # Draw confidence bar
            bar_width = int(100 * self.emotion_confidence)
            cv2.rectangle(image, 
                         (text_x, text_y + 10),
                         (text_x + bar_width, text_y + 20),
                         config['color'], -1)
            
            # Draw confidence percentage
            cv2.putText(image, confidence_text, (text_x + 110, text_y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['color'], 1)
                       
        except Exception as e:
            logger.error(f"Emotion overlay error: {e}")

    def get_emotion_data(self):
        """Get current emotion data with lightweight processing, including valence/arousal for frontend"""
        # Calculate valence/arousal from smoothed probabilities (like advanced processor)
        emotion_va_map = {
            'happy': (0.8, 0.6),
            'sad': (-0.7, -0.4),
            'angry': (-0.6, 0.7),
            'surprised': (0.2, 0.8),
            'fearful': (-0.5, 0.5),
            'disgusted': (-0.7, 0.2),
            'neutral': (0.0, 0.0)
        }
        smoothed_probs = self._emotion_ema.copy()
        total_weight = sum(smoothed_probs.values())
        valence = 0.0
        arousal = 0.0
        if total_weight > 0:
            valence = sum(prob * emotion_va_map.get(em, (0,0))[0] for em, prob in smoothed_probs.items()) / total_weight
            arousal = sum(prob * emotion_va_map.get(em, (0,0))[1] for em, prob in smoothed_probs.items()) / total_weight
        return {
            'emotion': self.current_emotion,
            'confidence': round(self.emotion_confidence, 3),
            'available_emotions': ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral'],
            'model_type': 'Fast-Geometric',
            'smoothing_frames': len(self.emotion_history),
            'valence': round(valence, 3),
            'arousal': round(arousal, 3),
            'smoothed_probabilities': {k: round(v, 3) for k, v in smoothed_probs.items()}
        }

    def process_frame(self, frame: np.ndarray):
        """
        Enhanced frame processing with parallel emotion detection
        """
        self.frame_count += 1
        
        # Simple MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # MediaPipe holistic processing
        results = self.holistic.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Store results (no locks - just assign)
        self.latest_results = results
        
        # Fast emotion detection directly in main thread (no lag)
        if results.face_landmarks:
            # Get emotion probabilities
            emotions = self.geometric_emotion.analyze_emotion(results.face_landmarks)
            
            # Debug: Log raw emotions every 30 frames
            if self.frame_count % 30 == 0:
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"Raw emotions: {sorted_emotions[:3]}")
            
            # Add to history for light smoothing
            self.emotion_history.append(emotions)
            if len(self.emotion_history) > self.max_history:
                self.emotion_history.pop(0)
            
            # EMA smoothing across frames
            for k, v in emotions.items():
                prev = self._emotion_ema.get(k, 0.0)
                self._emotion_ema[k] = (1 - self._emotion_alpha) * prev + self._emotion_alpha * v

            # Dominant with reduced hysteresis for faster switching
            import time as _time
            now = _time.time()
            best_emotion = max(self._emotion_ema.items(), key=lambda x: x[1])[0]
            best_conf = self._emotion_ema[best_emotion]

            # Switch if confidence is significantly higher or hold time elapsed
            if self.current_emotion != best_emotion:
                # Allow faster switching if new emotion is much stronger
                if (best_conf > self._emotion_ema.get(self.current_emotion, 0) + 0.15) or \
                   (now - self._emotion_last_switch_ts) >= self._emotion_min_hold_s:
                    logger.info(f"Emotion switched: {self.current_emotion} -> {best_emotion} ({best_conf:.2f})")
                    self.current_emotion = best_emotion
                    self._emotion_last_switch_ts = now
            else:
                # Update timestamp if confidence improves significantly
                if best_conf > self.emotion_confidence + 0.05:
                    self._emotion_last_switch_ts = now

            self.emotion_confidence = best_conf

        # NO FILTERING - MediaPipe's built-in smoothing is enough
        # Just draw the landmarks directly
        return self._draw_enhanced_landmarks(image, results, None)

    def _draw_enhanced_landmarks(self, image, results, hand_results=None):
        """
        GPU-accelerated dynamic face mesh + detailed hand meshes with responsive colors
        """
        try:
            # 1. DYNAMIC GPU-RENDERED FACE MESH with movement-responsive colors
            # Only draw default mesh when enabled; allows external custom overlays
            if results.face_landmarks:
                if self.use_mesh:
                    # Get dynamic colors based on face movement
                    mesh_color, contour_color = self._get_dynamic_face_colors(results.face_landmarks)

                    # Draw full face mesh tesselation (the actual mesh)
                    self.mp_drawing.draw_landmarks(
                        image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                        None,  # No landmark dots for cleaner mesh look
                        self.mp_drawing.DrawingSpec(color=mesh_color, thickness=1, circle_radius=1)
                    )

                    # Add face contours for definition and structure
                    self.mp_drawing.draw_landmarks(
                        image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                        self.mp_drawing.DrawingSpec(color=contour_color, thickness=1, circle_radius=1),
                        self.mp_drawing.DrawingSpec(color=contour_color, thickness=2)
                    )

                # Add emotion detection overlay regardless of mesh drawing
                self._draw_emotion_overlay(image, results.face_landmarks)
            
            # 2. Enhanced pose landmarks with GPU optimization
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
                )
            
            # 3. DETAILED HAND MESHES with full connection structure + palm mesh
            if results.left_hand_landmarks:
                # Draw standard hand connections
                self.mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0,200,0), thickness=2)
                )
                # Add enhanced palm mesh structure
                self._draw_palm_mesh_structure(image, results.left_hand_landmarks, (0,180,0))
                # Add fingertip highlights
                self._draw_fingertip_highlights(image, results.left_hand_landmarks, (0,255,255))  # Yellow for left
            
            if results.right_hand_landmarks:
                # Draw standard hand connections
                self.mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(200,100,100), thickness=2)
                )
                # Add enhanced palm mesh structure
                self._draw_palm_mesh_structure(image, results.right_hand_landmarks, (180,0,0))
                # Add fingertip highlights
                self._draw_fingertip_highlights(image, results.right_hand_landmarks, (255,0,255))  # Magenta for right
                
        except Exception as e:
            logger.error(f"GPU rendering error: {e}")

        return image

    def get_landmarks_for_classification(self):
        """Simple method to get latest landmarks - NO LOCKS"""
        return self.latest_results

    def get_latest_results(self):
        """Get the latest MediaPipe results for sign classification"""
        return self.latest_results
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance information"""
        return {
            "gpu_enabled": self.enable_gpu,
            "mesh_enabled": self.use_mesh,
            "frames_processed": self.frame_count,
            "filters_active": len(self.landmark_filters),
            "backend": "MediaPipe GPU" if self.enable_gpu else "MediaPipe CPU"
        }

    def close(self):
        # Stop advanced emotion processor
        if hasattr(self, 'advanced_emotion'):
            self.advanced_emotion.stop()
            
        if self.holistic:
            self.holistic.close()
        if self.hands:
            self.hands.close()