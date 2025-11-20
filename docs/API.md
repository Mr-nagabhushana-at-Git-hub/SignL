# SignL API Documentation

## Base URL
```
http://localhost:8000
```

## Core Endpoints

### Health Check
```http
GET /health
```
Returns server health status and GPU availability.

**Response:**
```json
{
    "status": "ok",
    "mediapipe_backend": "MediaPipe GPU",
    "gpu": "cuda:0",
    "cuda": true
}
```

### System Status
```http
GET /
```
Returns comprehensive system information.

**Response:**
```json
{
    "project": "SignL",
    "status": "🔥 GPU-ACCELERATED FACE RECOGNITION + SIGN LANGUAGE + EMOTION DETECTION",
    "face_recognition": "✅ Active (5 faces)",
    "sign_language": "✅ PyTorch GPU (10 signs)",
    "emotion_detection": "✅ Active",
    "known_faces": 5,
    "available_signs": ["hello", "thanks", "yes", "no", ...],
    "performance": {...}
}
```

## Face Recognition API

### List Known Faces
```http
GET /faces
```
Returns list of known faces and information.

**Response:**
```json
{
    "known_faces": ["Aishwarya A", "Nraju", "Sahana GM", ...],
    "total_faces": 5,
    "fase_data_path": "/path/to/face_data",
    "cache_path": "/path/to/cache"
}
```

### Refresh Face Cache
```http
POST /faces/refresh
```
Reloads face encodings from disk.

**Response:**
```json
{
    "status": "success",
    "message": "Refreshed cache with 5 faces"
}
```

### Get Face Recognition Parameters
```http
GET /faces/params
```
Returns current face recognition thresholds.

**Response:**
```json
{
    "params": {
        "confirm_sim": 0.6,
        "sticky_sim": 0.5,
        "drop_sim": 0.3,
        "margin_min": 0.05
    },
    "total_faces": 5
}
```

### Update Face Recognition Parameters
```http
POST /faces/params
Content-Type: application/json

{
    "confirm_sim": 0.7,
    "sticky_sim": 0.6,
    "drop_sim": 0.4
}
```

### Test Face Recognition
```http
POST /faces/test
```
Tests face recognition with current frame.

## Sign Language API

### Get Sign Classifier Info
```http
GET /signs
```
Returns sign classifier information.

**Response:**
```json
{
    "model_loaded": true,
    "actions": ["hello", "thanks", "yes", "no", ...],
    "sequence_length": 30,
    "device": "cuda:0",
    "cuda_available": true
}
```

### Reset Sign Sequence
```http
POST /signs/reset
```
Resets the current sign recognition sequence.

### Set Confidence Threshold
```http
POST /signs/confidence/{threshold}
```
Sets the confidence threshold for sign predictions.

**Example:**
```http
POST /signs/confidence/0.8
```

## Emotion Detection API

### Get Emotion Status
```http
GET /emotion/status
```
Returns detailed emotion detection status.

**Response:**
```json
{
    "emotion_detection": "✅ Advanced Geometric + Smoothing",
    "current_emotion": "happy",
    "confidence": 0.85,
    "valence": 0.7,
    "arousal": 0.5,
    "raw_probabilities": {...},
    "smoothed_probabilities": {...}
}
```

### Fine-tune Emotion Detection
```http
POST /emotion/tune
Content-Type: application/json

{
    "emotion_thresholds": {
        "happy": 0.6,
        "sad": 0.4,
        "angry": 0.5
    },
    "confidence_threshold": 0.45
}
```

## Gender Detection API

### Get Gender Detection Status
```http
GET /gender/status
```
Returns whether gender detection is enabled.

### Toggle Gender Detection
```http
POST /gender/toggle
```
Toggles gender detection on/off.

### Enable Gender Detection
```http
POST /gender/enable
```

### Disable Gender Detection
```http
POST /gender/disable
```

## WebSocket API

### Real-time Video Stream
```websocket
WS /ws
```

**Client Sends:** Raw video frame bytes (JPEG encoded)

**Server Sends:** JSON payload with processed results

**Payload Structure:**
```json
{
    "image": "base64_encoded_processed_frame",
    "faces": [
        {
            "box": [x1, y1, x2, y2],
            "name": "Person Name",
            "confidence": 0.95,
            "gender": "Male"
        }
    ],
    "emotion": {
        "emotion": "happy",
        "confidence": 0.85,
        "valence": 0.7,
        "arousal": 0.5
    },
    "sign": {
        "predicted_sign": "hello",
        "confidence": 0.92,
        "sequence_progress": 1.0
    },
    "debug": {
        "frame": 123,
        "mediapipe_ms": 15.2,
        "face_recognition_ms": 45.8,
        "sign_classification_ms": 28.3
    }
}
```

## Static Files

### Web Interface
```
GET /static/index.html
```
Main web application interface.

### Test Interface
```
GET /static/test.html
```
Testing interface.

## Interactive API Documentation

FastAPI provides interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Error Responses

All endpoints return standard HTTP status codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

Error response format:
```json
{
    "detail": "Error message here"
}
```

## Rate Limiting

No rate limiting is currently implemented for local development.
Consider adding rate limiting for production deployments.

## Authentication

No authentication is currently required.
Consider adding authentication for production deployments.
