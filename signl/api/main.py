# src/majorSignL/api/main.py
# Face Recognition Version - No More Corrupted Models!

import sys
from pathlib import Path
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
import logging
import cv2
import numpy as np
import base64
from typing import List
import time

from signl.utils.mediapipe_processor import MediaPipeProcessor
from signl.models.face_processor import FaceProcessor
from signl.models.pytorch_face_recognizer import PyTorchFaceRecognizer
from signl.models.sign_classifier import SignClassifier
from signl.models.gesture_sign_classifier import GestureSignClassifier
from signl.models.gender_processor import GenderProcessor
from signl.utils.video_processor import VideoProcessor
from signl.config import MODELS_DIR, SIGN_LANGUAGE_MODEL, FRONTEND_DIR
from signl.api.websocket_handler import WebSocketManager, websocket_endpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



app = FastAPI()

class AppState:
    def __init__(self):
        self.websocket_manager = WebSocketManager()
        self.latest_frame: np.ndarray = None
        self.is_processing = False
        # Cache for most recent face results to avoid empty frames between runs
        self.last_faces: List[dict] = []
        self.last_faces_ts: float = 0.0
        # Feature toggles
        self.gender_enabled: bool = True
        
        # Initialize video processor for test videos
        from pathlib import Path
        temp_video_dir = Path(__file__).resolve().parent.parent / "data" / "temp_videos"
        self.video_processor = VideoProcessor(temp_video_dir)
        
        # Initialize Enhanced MediaPipe with stable settings
        logger.info("🎯 Initializing MediaPipe...")
        # Keep internal mesh drawing off; we will draw custom overlays in this app
        self.mp_processor = MediaPipeProcessor(enable_gpu=True, use_mesh=False)
        logger.info("✅ MediaPipe ready!")
        
    # Initialize Face Recognition with MediaPipe Landmarks
        logger.info("👤 Initializing Face Recognition...")
        
        try:
            # Use new MediaPipe-based face recognizer instead of traditional face_recognition
            self.face_processor = PyTorchFaceRecognizer(MODELS_DIR)
            face_info = self.face_processor.get_known_faces_info()
            logger.info(f"✅ Face Recognition ready! Known faces: {face_info['total_faces']}")
            logger.info(f"📁 Fase data path: {face_info['fase_data_path']}")
            logger.info(f"💾 Cache path: {face_info['cache_path']}")
            logger.info(f"🔬 Method: {face_info['method']} with {face_info['key_landmarks_count']} key landmarks")
            if face_info['names']:
                logger.info(f"👥 Loaded faces: {', '.join(face_info['names'])}")
        except Exception as e:
            logger.error(f"❌ Face Recognition failed: {e}")
            # Fallback to traditional face recognition
            logger.info("🔄 Falling back to traditional face recognition...")
            try:
                self.face_processor = FaceProcessor(MODELS_DIR)
                face_info = self.face_processor.get_known_faces_info()
                logger.info(f"✅ Fallback Face Recognition ready! Known faces: {face_info['total_faces']}")
            except Exception as fallback_e:
                logger.error(f"❌ Fallback Face Recognition also failed: {fallback_e}")
                self.face_processor = None

        # Initialize Gender Detection
        logger.info("♀️♂️ Initializing Gender Detection...")
        try:
            self.gender_processor = GenderProcessor()
        except Exception as e:
            logger.error(f"❌ Gender Detection failed: {e}")
            self.gender_processor = None
        
        # Initialize Sign Language Classifier with GPU acceleration
        logger.info("🤟 Initializing Sign Language Classifier...")
        try:
            # Try PyTorch transformer model first
            self.sign_classifier = SignClassifier(SIGN_LANGUAGE_MODEL if SIGN_LANGUAGE_MODEL.exists() else None)
            
            # GPU warm-up for faster inference
            self.sign_classifier.warm_up_gpu()
            
            sign_info = self.sign_classifier.get_model_info()
            logger.info(f"✅ Sign Classifier ready! Model loaded: {sign_info['model_loaded']}")
            logger.info(f"🔥 GPU Device: {sign_info['device']}")
            logger.info(f"🤟 Available signs: {sign_info['actions']}")
            
            # Initialize gesture-based classifier as backup/alternative
            logger.info("🤲 Initializing Gesture-based Sign Classifier...")
            self.gesture_classifier = GestureSignClassifier(confidence_threshold=0.6)
            gesture_info = self.gesture_classifier.get_model_info()
            logger.info(f"✅ Gesture Classifier ready! Available signs: {len(gesture_info['actions'])}")
            
        except Exception as e:
            logger.error(f"❌ Sign Classifier failed: {e}")
            self.sign_classifier = None
            # Fallback to gesture classifier only
            try:
                logger.info("🔄 Falling back to gesture-based classifier only...")
                self.gesture_classifier = GestureSignClassifier(confidence_threshold=0.6)
                gesture_info = self.gesture_classifier.get_model_info()
                logger.info(f"✅ Gesture Classifier ready! Available signs: {len(gesture_info['actions'])}")
            except Exception as gesture_e:
                logger.error(f"❌ Gesture Classifier also failed: {gesture_e}")
                self.gesture_classifier = None

app.state.app_state = AppState()

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

async def ai_processing_task(app_state: AppState):
    logger.info("🚀 AI Processing Task - Face Recognition Mode")
    frame_count = 0
    last_time = time.time()
    
    while True:
        if app_state.latest_frame is not None and not app_state.is_processing:
            app_state.is_processing = True
            
            try:
                start_time = time.time()
                frame_copy = app_state.latest_frame.copy()
                
                # MediaPipe processing
                processed_frame = app_state.mp_processor.process_frame(frame_copy)
                mp_time = (time.time() - start_time) * 1000
                
                # Face recognition
                face_data = []
                face_time = 0
                if app_state.face_processor and frame_count % 2 == 0:
                    face_start = time.time()
                    try:
                        face_data = await asyncio.to_thread(app_state.face_processor.process_frame, frame_copy)
                        # Augment with gender when available
                        if app_state.gender_enabled and face_data and app_state.gender_processor is not None:
                            for f in face_data:
                                box = f.get("box")
                                if box and len(box) == 4:
                                    left, top, right, bottom = box
                                    # Clamp box to frame
                                    h, w = frame_copy.shape[:2]
                                    left = max(0, min(left, w-1))
                                    right = max(0, min(right, w-1))
                                    top = max(0, min(top, h-1))
                                    bottom = max(0, min(bottom, h-1))
                                    if right > left and bottom > top:
                                        face_crop = frame_copy[top:bottom, left:right]
                                        try:
                                            gender = await asyncio.to_thread(app_state.gender_processor.predict_gender, face_crop)
                                        except Exception:
                                            gender = "Unknown"
                                        f["gender"] = gender
                    except Exception as face_e:
                        logger.error(f"Face recognition error: {face_e}")
                    finally:
                        face_time = (time.time() - face_start) * 1000

                # Maintain face cache
                now_ts = time.time()
                if face_data:
                    app_state.last_faces = face_data
                    app_state.last_faces_ts = now_ts
                face_payload = app_state.last_faces if (now_ts - app_state.last_faces_ts) < 0.6 else []

                # Sign language classification
                sign_data = {"predicted_sign": "No Model", "confidence": 0.0}
                sign_time = 0
                
                # Try gesture-based classifier first (faster and doesn't need trained model)
                if app_state.gesture_classifier and frame_count % 2 == 0:
                    mp_results = app_state.mp_processor.get_landmarks_for_classification()
                    if mp_results:
                        sign_start = time.time()
                        try:
                            gesture_result = await asyncio.to_thread(app_state.gesture_classifier.process_frame, mp_results)
                            if gesture_result and gesture_result.get('confidence', 0) > 0.5:
                                sign_data = gesture_result
                            sign_time = (time.time() - sign_start) * 1000
                        except Exception as gesture_e:
                            logger.error(f"Gesture classification error: {gesture_e}")
                
                # Fallback to transformer model if available and gesture didn't detect
                if (sign_data.get('predicted_sign') == 'No sign detected' and 
                    app_state.sign_classifier and frame_count % 3 == 0):
                    mp_results = app_state.mp_processor.get_landmarks_for_classification()
                    if mp_results:
                        sign_start = time.time()
                        try:
                            transformer_result = await asyncio.to_thread(app_state.sign_classifier.update_sequence, mp_results)
                            if transformer_result and transformer_result.get('confidence', 0) > sign_data.get('confidence', 0):
                                sign_data = transformer_result
                            sign_time = (time.time() - sign_start) * 1000
                        except Exception as sign_e:
                            logger.error(f"Transformer classification error: {sign_e}")
                        finally:
                            if sign_time == 0:
                                sign_time = (time.time() - sign_start) * 1000
                
                # Drawing and payload creation: replace rectangles with facemesh-style outline colored by gender
                # We will draw a translucent face polygon (approx by convex hull of landmarks) and label above head
                mp_results = app_state.mp_processor.get_latest_results()
                h_img, w_img = processed_frame.shape[:2]
                def get_gender_color(g: str):
                    # BGR colors: Unknown=Green, Male=Blue, Female=Pink
                    if g == "Male":
                        return (255, 0, 0)
                    if g == "Female":
                        return (203, 192, 255)
                    return (0, 255, 0)

                def draw_face_overlay(landmarks, color, name_label: str):
                    pts = []
                    for lm in landmarks:
                        x = int(lm.x * w_img)
                        y = int(lm.y * h_img)
                        pts.append([x, y])
                    if len(pts) < 3:
                        return
                    pts_np = np.array(pts, dtype=np.int32)
                    hull = cv2.convexHull(pts_np)
                    # Translucent fill
                    overlay = processed_frame.copy()
                    cv2.fillConvexPoly(overlay, hull, (int(color[0]*0.5), int(color[1]*0.5), int(color[2]*0.5)))
                    processed_frame[:] = cv2.addWeighted(overlay, 0.25, processed_frame, 0.75, 0)
                    # Outline
                    cv2.polylines(processed_frame, [hull], isClosed=True, color=color, thickness=2)
                    # Label above top-most point
                    top_idx = np.argmin(hull[:, 0, 1])
                    top_pt = hull[top_idx, 0]
                    cv2.putText(processed_frame, name_label, (int(top_pt[0]) - 20, max(20, int(top_pt[1]) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Map current MediaPipe landmarks to the best matching detected face box and draw once
                if mp_results and getattr(mp_results, 'face_landmarks', None):
                    face_landmarks = mp_results.face_landmarks.landmark

                    # Compute landmarks bounding box
                    xs = [int(lm.x * w_img) for lm in face_landmarks]
                    ys = [int(lm.y * h_img) for lm in face_landmarks]
                    lm_left, lm_right = max(min(xs), 0), min(max(xs), w_img - 1)
                    lm_top, lm_bottom = max(min(ys), 0), min(max(ys), h_img - 1)

                    def iou(a, b):
                        ax1, ay1, ax2, ay2 = a
                        bx1, by1, bx2, by2 = b
                        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                        inter = iw * ih
                        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                        union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1
                        return inter / union

                    chosen_name = "Unknown"
                    chosen_gender = "Unknown"
                    if face_payload:
                        lm_box = (lm_left, lm_top, lm_right, lm_bottom)
                        best = None
                        best_iou = -1.0
                        for f in face_payload:
                            box = f.get("box")
                            if not box or len(box) != 4:
                                continue
                            score = iou((box[0], box[1], box[2], box[3]), lm_box)
                            if score > best_iou:
                                best = f
                                best_iou = score
                        if best is not None:
                            chosen_name = best.get("name", "Unknown")
                            if app_state.gender_enabled:
                                chosen_gender = best.get("gender", "Unknown")

                    color = get_gender_color(chosen_gender)
                    label = f"{chosen_name} ({chosen_gender})" if app_state.gender_enabled else chosen_name
                    # Draw gender-colored outer outline + translucent fill (convex hull)
                    draw_face_overlay(face_landmarks, color, label)

                    # Draw semi-transparent tessellation lines for detailed face mesh
                    try:
                        proc = app_state.mp_processor
                        mp_draw = proc.mp_drawing
                        mp_hol = proc.mp_holistic
                        mesh_overlay = processed_frame.copy()
                        # Light gray lines; no landmark points, only connections
                        mp_draw.draw_landmarks(
                            mesh_overlay,
                            mp_results.face_landmarks,
                            mp_hol.FACEMESH_TESSELATION,
                            None,
                            mp_draw.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
                        )
                        # Blend overlay to make lines transparent but visible
                        processed_frame[:] = cv2.addWeighted(mesh_overlay, 0.25, processed_frame, 0.75, 0)
                    except Exception as _e:
                        # Non-fatal drawing issue; continue
                        pass

                cv2.putText(processed_frame, f"MediaPipe: {mp_time:.1f}ms | Faces: {len(face_payload)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Sign: {sign_data.get('predicted_sign', 'N/A')} ({sign_data.get('confidence', 0):.2f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(processed_frame, f"Frame: #{frame_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                emotion_data = app_state.mp_processor.get_emotion_data()
                
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                payload = {
                    "image": jpg_as_text,
                    "faces": face_payload,
                    "toggles": {"gender_enabled": app_state.gender_enabled},
                    "emotion": emotion_data,
                    "sign": sign_data,
                    "debug": {
                        "frame": frame_count,
                        "mediapipe_ms": round(mp_time, 1),
                        "face_recognition_ms": round(face_time, 1),
                        "sign_classification_ms": round(sign_time, 1),
                        "faces_recognized": len([f for f in face_payload if f.get('name') != 'Unknown']),
                        "predicted_sign": sign_data.get('predicted_sign', 'N/A'),
                        "sign_confidence": round(sign_data.get('confidence', 0), 3),
                        "current_emotion": emotion_data.get('emotion', 'neutral'),
                        "emotion_confidence": emotion_data.get('confidence', 0.0)
                    }
                }
                
                await app_state.websocket_manager.broadcast_json(payload)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - last_time)
                    recognized_count = len([f for f in face_payload if f.get('name') != 'Unknown'])
                    current_sign = sign_data.get('predicted_sign', 'N/A')
                    logger.info(f"🔥 Performance: {fps:.1f} FPS | Faces: {len(face_payload)} | Recognized: {recognized_count} | Sign: {current_sign}")
                    last_time = time.time()
                    
            except Exception as e:
                logger.error(f"Main processing loop error: {e}")
            finally:
                app_state.is_processing = False
        
        await asyncio.sleep(0.03)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ai_processing_task(app.state.app_state))
    logger.info("🎯 Project Synapse - Face Recognition Mode Active")

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket_endpoint(websocket, app.state.app_state)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Landing page with links to web interface and API"""
    face_count = len(app.state.app_state.face_processor.known_face_names) if app.state.app_state.face_processor else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SignL - AI System</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                max-width: 600px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }}
            h1 {{
                font-size: 2.5em;
                margin: 0 0 10px 0;
                text-align: center;
            }}
            .subtitle {{
                text-align: center;
                font-size: 1.1em;
                margin-bottom: 30px;
                opacity: 0.9;
            }}
            .status {{
                background: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            }}
            .status-item {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .status-item:last-child {{
                border-bottom: none;
            }}
            .links {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-top: 30px;
            }}
            .link-button {{
                background: rgba(255, 255, 255, 0.9);
                color: #667eea;
                padding: 15px 25px;
                border-radius: 10px;
                text-decoration: none;
                text-align: center;
                font-weight: bold;
                transition: all 0.3s;
                display: block;
            }}
            .link-button:hover {{
                background: white;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }}
            .link-button.primary {{
                grid-column: 1 / -1;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 1.2em;
            }}
            .emoji {{
                font-size: 1.2em;
                margin-right: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 SignL</h1>
            <div class="subtitle">Real-Time Sign Language & Face Recognition System</div>
            
            <div class="status">
                <div class="status-item">
                    <span><span class="emoji">👤</span>Face Recognition</span>
                    <span><strong>{face_count} people loaded</strong></span>
                </div>
                <div class="status-item">
                    <span><span class="emoji">🤟</span>Sign Language</span>
                    <span><strong>10 signs available</strong></span>
                </div>
                <div class="status-item">
                    <span><span class="emoji">😊</span>Emotion Detection</span>
                    <span><strong>7 emotions</strong></span>
                </div>
                <div class="status-item">
                    <span><span class="emoji">💻</span>Processing Mode</span>
                    <span><strong>CPU (4 cores)</strong></span>
                </div>
            </div>
            
            <div class="links">
                <a href="/static/index.html" class="link-button primary">
                    <span class="emoji">🎥</span>Launch Web Interface
                </a>
                <a href="/docs" class="link-button">
                    <span class="emoji">📚</span>API Docs
                </a>
                <a href="/health" class="link-button">
                    <span class="emoji">❤️</span>Health Check
                </a>
                <a href="/faces" class="link-button">
                    <span class="emoji">👥</span>View Faces
                </a>
                <a href="/emotion/status" class="link-button">
                    <span class="emoji">🎭</span>Emotions
                </a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/api/status")
async def api_status():
    """API endpoint for system status (JSON)"""
    face_count = len(app.state.app_state.face_processor.known_face_names) if app.state.app_state.face_processor else 0
    sign_info = app.state.app_state.sign_classifier.get_model_info() if app.state.app_state.sign_classifier else {"model_loaded": False, "actions": [], "device": "N/A"}
    mp_info = app.state.app_state.mp_processor.get_performance_info()
    emotion_info = app.state.app_state.mp_processor.get_emotion_data()
    
    return {
        "project": "MajorSignL",
        "status": "🔥 GPU-ACCELERATED FACE RECOGNITION + SIGN LANGUAGE + EMOTION DETECTION",
        "mediapipe": f"✅ {mp_info['backend']} (Mesh: {'ON' if mp_info['mesh_enabled'] else 'OFF'})",
        "face_recognition": f"✅ Active ({face_count} faces)" if app.state.app_state.face_processor else "❌ Inactive",
        "emotion_detection": f"✅ Active - Current: {emotion_info.get('emotion', 'neutral')} ({emotion_info.get('confidence', 0):.1%})",
        "sign_language": f"✅ PyTorch GPU ({len(sign_info['actions'])} signs)" if sign_info["model_loaded"] else "❌ Model Not Found",
        "gpu_device": sign_info.get("device", "N/A"),
        "cuda_available": sign_info.get("cuda_available", False),
        "gpu_name": sign_info.get("gpu_name", "N/A"),
        "frames_processed": mp_info.get("frames_processed", 0),
        "known_faces": face_count,
        "available_signs": sign_info["actions"],
        "available_emotions": emotion_info.get('available_emotions', []),
        "current_emotion": emotion_info.get('emotion', 'neutral'),
        "performance": {
            "mediapipe_filters": mp_info.get("filters_active", 0),
            "sign_predictions": sign_info.get("total_predictions", 0),
            "last_prediction_ms": sign_info.get("last_prediction_time_ms", 0),
            "emotion_confidence": emotion_info.get('confidence', 0.0)
        }
    }

@app.get("/health")
async def health_check():
    """Simple health endpoint used by the frontend connectivity test"""
    try:
        sign_info = app.state.app_state.sign_classifier.get_model_info() if app.state.app_state.sign_classifier else {"model_loaded": False}
        mp_info = app.state.app_state.mp_processor.get_performance_info()
        return {
            "status": "ok",
            "mediapipe_backend": mp_info.get("backend"),
            "gpu": sign_info.get("device", "CPU"),
            "cuda": sign_info.get("cuda_available", False)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/gender/status")
async def gender_status():
    return {"enabled": app.state.app_state.gender_enabled}

@app.post("/gender/toggle")
async def gender_toggle():
    app.state.app_state.gender_enabled = not app.state.app_state.gender_enabled
    return {"enabled": app.state.app_state.gender_enabled}

@app.post("/gender/enable")
async def gender_enable():
    app.state.app_state.gender_enabled = True
    return {"enabled": True}

@app.post("/gender/disable")
async def gender_disable():
    app.state.app_state.gender_enabled = False
    return {"enabled": False}

@app.get("/emotion/status")
async def get_emotion_status():
    """Get detailed emotion detection status and parameters"""
    emotion_data = app.state.app_state.mp_processor.get_emotion_data()
    return {
        "emotion_detection": "✅ Advanced Geometric + Smoothing + Valence-Arousal",
        "current_emotion": emotion_data.get('emotion', 'neutral'),
        "confidence": emotion_data.get('confidence', 0.0),
        "valence": emotion_data.get('valence', 0.0),  # -1 to +1
        "arousal": emotion_data.get('arousal', 0.0),  # -1 to +1
        "raw_probabilities": emotion_data.get('raw_probabilities', {}),
        "smoothed_probabilities": emotion_data.get('smoothed_probabilities', {}),
        "fine_tune_params": emotion_data.get('fine_tune_params', {}),
        "performance": {
            "total_processed": emotion_data.get('total_processed', 0),
            "avg_processing_ms": emotion_data.get('avg_processing_time_ms', 0)
        }
    }

@app.post("/emotion/tune")
async def tune_emotion_detection(tune_params: dict):
    """
    Fine-tune emotion detection parameters for lightweight geometric emotion detection
    
    Example:
    {
        "emotion_thresholds": {
            "happy": 0.6,
            "sad": 0.4,
            "neutral": 0.3
        },
        "confidence_threshold": 0.45
    }
    """
    try:
        # Use the geometric emotion detector from MediaPipe processor
        emotion_detector = app.state.app_state.mp_processor.geometric_emotion

        emotion_thresholds = tune_params.get('emotion_thresholds', {})
        confidence_threshold = tune_params.get('confidence_threshold', 0.5)

        # Update the emotion detector thresholds
        if hasattr(emotion_detector, 'update_thresholds'):
            emotion_detector.update_thresholds(emotion_thresholds, confidence_threshold)
        else:
            # Simple threshold update for emotion detector
            for emotion, threshold in emotion_thresholds.items():
                if emotion in emotion_detector.emotion_thresholds:
                    emotion_detector.emotion_thresholds[emotion] = threshold
            emotion_detector.confidence_threshold = confidence_threshold

        return {
            "success": True,
            "message": "Emotion detection parameters updated",
            "updated_params": tune_params
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
@app.get("/faces")
async def list_known_faces():
    if app.state.app_state.face_processor:
        face_info = app.state.app_state.face_processor.get_known_faces_info()
        return {
            "known_faces": face_info['names'],
            "total_faces": face_info['total_faces'],
            "fase_data_path": face_info['fase_data_path'],
            "cache_path": face_info['cache_path']
        }
    return {"known_faces": [], "total_faces": 0}

@app.post("/faces/refresh")
async def refresh_face_cache():
    """Refresh the face recognition cache"""
    if app.state.app_state.face_processor:
        try:
            count = app.state.app_state.face_processor.refresh_cache()
            return {"status": "success", "message": f"Refreshed cache with {count} faces"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Face processor not available"}

@app.get("/faces/params")
async def get_face_params():
    """Get current face recognition thresholds and gating parameters"""
    if app.state.app_state.face_processor and hasattr(app.state.app_state.face_processor, 'get_known_faces_info'):
        info = app.state.app_state.face_processor.get_known_faces_info()
        return {
            "params": info.get("params", {}),
            "total_faces": info.get("total_faces", 0),
            "names": info.get("names", [])
        }
    return {"params": {}, "total_faces": 0, "names": []}

@app.post("/faces/params")
async def update_face_params(params: dict):
    """Update face recognition parameters (confirm_sim, sticky_sim, drop_sim, margin_min, switch_cooldown_s, required_consecutive, iou_threshold)"""
    if app.state.app_state.face_processor and hasattr(app.state.app_state.face_processor, 'update_parameters'):
        try:
            updated = app.state.app_state.face_processor.update_parameters(params)
            return {"status": "success", "params": updated}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Face processor not available or not configurable"}

@app.post("/faces/test")
async def test_face_recognition():
    """Test face recognition with current frame or synthetic test"""
    if not app.state.app_state.face_processor:
        return {"status": "error", "message": "Face processor not available"}

    try:
        # Use the latest frame if available, otherwise create a test pattern
        if app.state.app_state.latest_frame is not None:
            test_frame = app.state.app_state.latest_frame.copy()
            test_type = "latest_frame"
        else:
            # Create a simple test pattern (white square in center to simulate face)
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add a white square in center to simulate face detection
            test_frame[200:280, 280:360] = [255, 255, 255]
            test_type = "synthetic_pattern"

        # Test face recognition
        results = app.state.app_state.face_processor.process_frame(test_frame)

        return {
            "status": "success",
            "test_type": test_type,
            "results": results,
            "known_faces": len(app.state.app_state.face_processor.known_face_names),
            "feature_count": len(app.state.app_state.face_processor.known_face_features),
            "message": f"Tested with {test_type}. Found {len(results)} faces."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/faces/debug")
async def debug_faces():
    """Debug endpoint to see current face recognition state"""
    if not app.state.app_state.face_processor:
        return {"error": "Face processor not available"}
    
    # Get current face info
    face_info = app.state.app_state.face_processor.get_known_faces_info()
    
    # Get current last faces cache
    last_faces = app.state.app_state.last_faces or []
    
    return {
        "known_faces": face_info.get("names", []),
        "total_faces": face_info.get("total_faces", 0),
        "feature_vector_size": face_info.get("feature_vector_size", 0),
        "params": face_info.get("params", {}),
        "last_faces_cache": last_faces,
        "cache_timestamp": app.state.app_state.last_faces_ts,
        "cache_age_seconds": time.time() - app.state.app_state.last_faces_ts if app.state.app_state.last_faces_ts > 0 else None
    }

@app.get("/debug/face-paths")
async def debug_face_paths():
    """Debug endpoint to check face data paths"""
    from signl.config import FACE_DATA_DIR, PROJECT_ROOT
    
    debug_info = {
        "project_root": str(PROJECT_ROOT),
        "fase_data_path": str(FACE_DATA_DIR),
        "fase_data_exists": FACE_DATA_DIR.exists(),
        "person_folders": []
    }
    
    if FACE_DATA_DIR.exists():
        for item in FACE_DATA_DIR.iterdir():
            if item.is_dir() and not item.name.endswith('.pkl'):
                image_count = len(list(item.glob('*.jpg')) + list(item.glob('*.jpeg')) + 
                                 list(item.glob('*.png')) + list(item.glob('*.JPG')) + 
                                 list(item.glob('*.JPEG')) + list(item.glob('*.PNG')))
                debug_info["person_folders"].append({
                    "name": item.name,
                    "image_count": image_count
                })
    
    return debug_info

@app.get("/signs")
async def get_sign_info():
    """Get sign language classifier information"""
    info = {
        "gesture_classifier": None,
        "transformer_classifier": None,
        "active_method": "none"
    }
    
    if app.state.app_state.gesture_classifier:
        info["gesture_classifier"] = app.state.app_state.gesture_classifier.get_model_info()
        info["active_method"] = "gesture"
    
    if app.state.app_state.sign_classifier:
        info["transformer_classifier"] = app.state.app_state.sign_classifier.get_model_info()
        if info["transformer_classifier"].get("model_loaded"):
            info["active_method"] = "both"
    
    return info

@app.post("/signs/reset")
async def reset_sign_sequence():
    """Reset the current sign sequence"""
    reset_count = 0
    if app.state.app_state.sign_classifier:
        app.state.app_state.sign_classifier.reset_sequence()
        reset_count += 1
    if app.state.app_state.gesture_classifier:
        app.state.app_state.gesture_classifier.reset_sequence()
        reset_count += 1
    
    return {"status": "success", "message": f"Reset {reset_count} classifier(s)"}

@app.post("/signs/confidence/{threshold}")
async def set_confidence_threshold(threshold: float):
    """Set the confidence threshold for sign predictions"""
    set_count = 0
    if app.state.app_state.sign_classifier:
        app.state.app_state.sign_classifier.set_confidence_threshold(threshold)
        set_count += 1
    if app.state.app_state.gesture_classifier:
        app.state.app_state.gesture_classifier.set_confidence_threshold(threshold)
        set_count += 1
    
    return {"status": "success", "message": f"Confidence threshold set to {threshold} for {set_count} classifier(s)"}

@app.post("/video/upload")
async def upload_test_video(file: UploadFile = File(...)):
    """
    Upload a test video for sign language recognition testing
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            return {
                "status": "error",
                "message": f"Invalid file type: {file.content_type}. Please upload a video file."
            }
        
        # Read file data
        video_data = await file.read()
        
        # Save video
        video_path = app.state.app_state.video_processor.save_uploaded_video(
            video_data, 
            file.filename or 'test_video.mp4'
        )
        
        return {
            "status": "success",
            "message": f"Video uploaded successfully: {file.filename}",
            "filename": file.filename,
            "size_bytes": len(video_data),
            "video_path": str(video_path)
        }
        
    except Exception as e:
        logger.error(f"Video upload error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/video/process/{filename}")
async def process_test_video(filename: str):
    """
    Process uploaded test video and return sign language predictions
    """
    try:
        video_path = app.state.app_state.video_processor.temp_dir / filename
        
        if not video_path.exists():
            return {
                "status": "error",
                "message": f"Video not found: {filename}"
            }
        
        # Use gesture classifier for processing
        classifier = app.state.app_state.gesture_classifier
        if not classifier:
            # Fallback to transformer classifier
            classifier = app.state.app_state.sign_classifier
        
        if not classifier:
            return {
                "status": "error",
                "message": "No sign language classifier available"
            }
        
        # Process video
        result = await app.state.app_state.video_processor.process_video(
            video_path,
            app.state.app_state.mp_processor,
            classifier
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/video/list")
async def list_test_videos():
    """List all uploaded test videos"""
    try:
        temp_dir = app.state.app_state.video_processor.temp_dir
        videos = []
        
        if temp_dir.exists():
            for video_file in temp_dir.glob('*'):
                if video_file.is_file():
                    videos.append({
                        "filename": video_file.name,
                        "size_bytes": video_file.stat().st_size,
                        "modified": video_file.stat().st_mtime
                    })
        
        return {
            "status": "success",
            "videos": videos,
            "count": len(videos)
        }
        
    except Exception as e:
        logger.error(f"Video list error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.delete("/video/{filename}")
async def delete_test_video(filename: str):
    """Delete a test video"""
    try:
        video_path = app.state.app_state.video_processor.temp_dir / filename
        
        if video_path.exists():
            video_path.unlink()
            return {
                "status": "success",
                "message": f"Video deleted: {filename}"
            }
        else:
            return {
                "status": "error",
                "message": f"Video not found: {filename}"
            }
            
    except Exception as e:
        logger.error(f"Video deletion error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
