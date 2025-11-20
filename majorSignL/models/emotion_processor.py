# src/majorSignL/models/emotion_processor.py

import cv2
import numpy as np
import torch
import threading
import queue
import time
from typing import Dict, Optional, Any
import logging
from hsemotion.facial_emotions import HSEmotionRecognizer
from PIL import Image

logger = logging.getLogger(__name__)

class ParallelEmotionProcessor:
    """
    High-accuracy emotion detection using HSEmotion with parallel processing
    Runs in separate thread to avoid blocking main pipeline
    """
    
    def __init__(self, device: str = "cuda", model_name: str = "enet_b0_8_best_vgaf", use_hsemotion: bool = True):
        """
        Initialize parallel emotion processor
        
        Args:
            device: CUDA device for emotion detection
            model_name: HSEmotion model variant
            use_hsemotion: Whether to use HSEmotion or fallback to simple detection
        """
        self.device = device
        self.model_name = model_name
        self.use_hsemotion = use_hsemotion
        
        # Threading components
        self.input_queue = queue.Queue(maxsize=3)  # Small queue to avoid lag
        self.result_queue = queue.Queue(maxsize=1)
        self.processing_thread = None
        self.is_running = False
        
        # Current emotion state
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.emotion_probabilities = {}
        self.last_update_time = 0
        
        # HSEmotion model (initialized in thread)
        self.emotion_model = None
        self.model_initialized = False
        
        # Performance tracking
        self.total_processed = 0
        self.avg_processing_time = 0
        
        logger.info(f"🎭 Initializing Parallel Emotion Processor - Device: {device}")
        
    def start(self):
        """Start the parallel emotion processing thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("🎭 Parallel emotion processing thread started")
        
    def stop(self):
        """Stop the parallel emotion processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        logger.info("🎭 Parallel emotion processing stopped")
        
    def process_frame_async(self, frame: np.ndarray, face_box: Optional[tuple] = None):
        """
        Queue frame for asynchronous emotion processing
        
        Args:
            frame: Input frame
            face_box: Optional face bounding box (x1, y1, x2, y2)
        """
        try:
            # Add frame to queue (non-blocking)
            frame_data = {
                'frame': frame.copy(),
                'face_box': face_box,
                'timestamp': time.time()
            }
            
            # Replace old frame if queue is full (keep only latest)
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()  # Remove old frame
                except queue.Empty:
                    pass
                    
            self.input_queue.put_nowait(frame_data)
            
        except queue.Full:
            pass  # Skip if queue is full - we want real-time performance
            
    def get_emotion_data(self) -> Dict[str, Any]:
        """
        Get current emotion detection results
        
        Returns:
            Dictionary with emotion data
        """
        # Try to get latest results (non-blocking)
        try:
            while True:  # Get the latest result
                try:
                    result = self.result_queue.get_nowait()
                    self.current_emotion = result['emotion']
                    self.emotion_confidence = result['confidence'] 
                    self.emotion_probabilities = result['probabilities']
                    self.last_update_time = result['timestamp']
                except queue.Empty:
                    break
        except:
            pass
            
        return {
            'emotion': self.current_emotion,
            'confidence': self.emotion_confidence,
            'probabilities': self.emotion_probabilities,
            'last_update': self.last_update_time,
            'model_loaded': self.model_initialized,
            'total_processed': self.total_processed,
            'avg_processing_time_ms': round(self.avg_processing_time * 1000, 1)
        }
        
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        logger.info("🎭 Starting emotion processing loop...")
        
        # Initialize HSEmotion model in the processing thread
        if self.use_hsemotion:
            try:
                logger.info(f"🎭 Loading HSEmotion model: {self.model_name}")
                self.emotion_model = HSEmotionRecognizer(
                    model_name=self.model_name,
                    device=self.device
                )
                self.model_initialized = True
                logger.info("🎭 HSEmotion model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load HSEmotion model: {e}")
                logger.info("🎭 Falling back to simple emotion detection")
                self.use_hsemotion = False
        
        if not self.use_hsemotion:
            self.model_initialized = True
            logger.info("🎭 Using simple geometric emotion detection")
            
        processing_times = []
        
        while self.is_running:
            try:
                # Get frame from queue (blocking with timeout)
                frame_data = self.input_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Extract face region if bounding box provided
                frame = frame_data['frame']
                face_box = frame_data.get('face_box')
                
                if face_box:
                    x1, y1, x2, y2 = face_box
                    face_region = frame[y1:y2, x1:x2]
                else:
                    face_region = frame
                    
                # Skip if face region is too small
                if face_region.size == 0 or min(face_region.shape[:2]) < 64:
                    continue
                
                if self.use_hsemotion:
                    # HSEmotion processing
                    try:
                        # Resize face region for consistent processing
                        face_resized = cv2.resize(face_region, (224, 224))
                        
                        # Convert BGR to RGB and create PIL Image
                        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(face_rgb.astype(np.uint8))
                        
                        # Run emotion detection 
                        emotion_scores = self.emotion_model.predict_emotions(pil_image)
                        
                        # Handle different return formats
                        if isinstance(emotion_scores, dict):
                            # Standard dictionary format
                            scores_dict = emotion_scores
                        else:
                            # If it returns a list or tensor, convert to dict
                            emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                            if len(emotion_scores) == 7:
                                scores_dict = dict(zip(emotion_names, emotion_scores))
                            else:
                                # Fallback - just use the first few emotions
                                scores_dict = {name: score for name, score in zip(emotion_names, emotion_scores[:7])}
                        
                    except Exception as model_error:
                        logger.error(f"❌ HSEmotion model error: {model_error}")
                        # Skip this frame and continue
                        continue
                else:
                    # Simple emotion detection based on basic image analysis
                    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    
                    # Simple heuristic-based emotion detection
                    # This is a placeholder - you could add more sophisticated logic
                    brightness = np.mean(gray)
                    contrast = np.std(gray)
                    
                    # Basic emotion mapping based on image properties
                    if brightness > 120 and contrast > 30:
                        scores_dict = {'happy': 0.7, 'neutral': 0.3, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}
                    elif brightness < 80:
                        scores_dict = {'sad': 0.6, 'neutral': 0.4, 'happy': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}
                    else:
                        scores_dict = {'neutral': 0.8, 'happy': 0.1, 'sad': 0.1, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}                # Get dominant emotion
                dominant_emotion = max(scores_dict, key=scores_dict.get)
                confidence = scores_dict[dominant_emotion]
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Keep only recent processing times for average
                if len(processing_times) > 30:
                    processing_times = processing_times[-30:]
                    
                self.avg_processing_time = np.mean(processing_times)
                self.total_processed += 1
                
                # Put result in queue (replace old result if queue is full)
                result = {
                    'emotion': dominant_emotion,
                    'confidence': confidence,
                    'probabilities': scores_dict,
                    'timestamp': time.time(),
                    'processing_time': processing_time
                }
                
                try:
                    # Remove old result if queue is full
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass
                    
            except queue.Empty:
                continue  # No frame to process, keep waiting
            except Exception as e:
                logger.error(f"❌ Emotion processing error: {e}")
                continue
                
        logger.info("🎭 Emotion processing loop stopped")
        
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'model_initialized': self.model_initialized,
            'total_processed': self.total_processed,
            'avg_processing_time_ms': round(self.avg_processing_time * 1000, 1),
            'queue_size': self.input_queue.qsize(),
            'device': self.device,
            'model_name': self.model_name,
            'thread_alive': self.processing_thread.is_alive() if self.processing_thread else False
        }

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop()
