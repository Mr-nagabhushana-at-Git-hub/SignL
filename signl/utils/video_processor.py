"""
Video processing module for testing sign language recognition with pre-recorded videos
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import base64
import time

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Process pre-recorded videos for sign language testing
    """
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.current_video = None
        self.frame_count = 0
        logger.info("✅ VideoProcessor initialized")
    
    def save_uploaded_video(self, video_data: bytes, filename: str) -> Path:
        """Save uploaded video to temp directory"""
        try:
            video_path = self.temp_dir / filename
            with open(video_path, 'wb') as f:
                f.write(video_data)
            
            self.current_video = video_path
            logger.info(f"✅ Video saved: {video_path}")
            return video_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save video: {e}")
            raise
    
    async def process_video(self, video_path: Path, mp_processor, sign_classifier) -> Dict:
        """
        Process video and extract sign language predictions
        NOTE: This requires opencv-python to be installed
        """
        try:
            # Import cv2 here to avoid import errors if not installed
            import cv2
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {
                    "status": "error",
                    "message": "Failed to open video file"
                }
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            predictions = []
            frame_idx = 0
            
            logger.info(f"Processing video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 3rd frame for efficiency
                if frame_idx % 3 == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with MediaPipe
                    mp_results = mp_processor.process_frame(frame_rgb)
                    
                    # Get sign prediction
                    if mp_results:
                        landmarks = mp_processor.get_landmarks_for_classification()
                        if landmarks:
                            sign_result = sign_classifier.process_frame(landmarks)
                            
                            if sign_result and sign_result.get('confidence', 0) > 0.5:
                                timestamp = frame_idx / fps if fps > 0 else 0
                                predictions.append({
                                    'frame': frame_idx,
                                    'timestamp': timestamp,
                                    'sign': sign_result.get('predicted_sign'),
                                    'confidence': sign_result.get('confidence')
                                })
                
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"Processing: {progress:.1f}% ({frame_idx}/{total_frames})")
            
            cap.release()
            
            # Analyze results
            unique_signs = {}
            for pred in predictions:
                sign = pred['sign']
                if sign not in unique_signs:
                    unique_signs[sign] = []
                unique_signs[sign].append(pred['confidence'])
            
            # Calculate average confidence per sign
            sign_summary = {}
            for sign, confidences in unique_signs.items():
                sign_summary[sign] = {
                    'count': len(confidences),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'max_confidence': max(confidences)
                }
            
            return {
                "status": "success",
                "video_info": {
                    "total_frames": total_frames,
                    "fps": fps,
                    "duration": duration,
                    "processed_frames": frame_idx
                },
                "predictions": predictions,
                "summary": {
                    "total_predictions": len(predictions),
                    "unique_signs": len(unique_signs),
                    "signs_detected": sign_summary
                }
            }
            
        except ImportError:
            logger.error("❌ opencv-python not installed - cannot process videos")
            return {
                "status": "error",
                "message": "Video processing requires opencv-python to be installed"
            }
        except Exception as e:
            logger.error(f"❌ Video processing error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                for file in self.temp_dir.glob('*'):
                    if file.is_file():
                        file.unlink()
                logger.info("✅ Temporary files cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def get_info(self) -> Dict:
        """Get processor information"""
        return {
            "temp_dir": str(self.temp_dir),
            "current_video": str(self.current_video) if self.current_video else None,
            "frame_count": self.frame_count
        }
