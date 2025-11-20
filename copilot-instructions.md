# Copilot Instructions for MajorSignL

## Project Overview
- **MajorSignL** is a real-time sign language and face recognition system using FastAPI, MediaPipe, and Transformer models.
- The system processes webcam video streams, recognizes sign language gestures, and identifies people using face recognition.
- Key components: API server (`src/majorSignL/api/`), face/sign processors (`src/majorSignL/models/`), MediaPipe utilities (`src/majorSignL/utils/`), and web frontend (`src/majorSignL/frontend/`).

## Architecture & Data Flow
- **Face data**: Images are organized in `src/data/fase_data/{Person Name}/` (10+ images per person). Encodings are cached in `src/data/face_cache/`.
- **Models**: Pretrained models are in `src/data/models/`. Sign language model is trained via `src/majorSignL/train_model.py`.
- **API**: FastAPI app in `src/majorSignL/api/main.py` exposes REST and WebSocket endpoints. Web frontend connects via WebSocket for real-time video.
- **Processing rates**: Face recognition runs every 5th frame, sign recognition every 2nd frame, MediaPipe every frame (see `main.py`).

## Developer Workflows
- **Environment**: Use `env.yml` with Conda/Mamba. Activate with `mamba activate SignL`.
- **Start server**: Use `start_server.ps1` (Windows) or `start_server.sh` (Linux/WSL2). Manual: `python -m uvicorn majorSignL.api.main:app --host 0.0.0.0 --port 8000 --reload` from `src/`.
- **Train sign model**: `python src/majorSignL/train_model.py` (requires training data).
- **Face cache**: Refresh with `POST /faces/refresh` or via API call.
- **Debugging**: Use `/debug/face-paths` endpoint to verify face data loading.

## Project-Specific Conventions
- **Face folders**: Each person must have a folder with 10+ images in `fase_data/`.
- **Frame skipping**: Processing intervals are hardcoded for performance (see `main.py`).
- **Model paths**: All models and caches are stored under `src/data/`.
- **Web frontend**: Static files in `src/majorSignL/frontend/`.

## Integration & Dependencies
- **MediaPipe**: Used for landmark detection (see `utils/mediapipe_processor.py`).
- **face_recognition**: For face encoding and matching.
- **PyTorch**: For sign language model (CUDA required for GPU acceleration).
- **WebSocket**: Real-time video streaming between browser and backend.

## Examples & Patterns
- To add a new person: create a folder in `src/data/fase_data/` with 10+ images, then refresh face cache.
- To adjust performance: change frame skip intervals in `main.py` and image resize in `face_processor.py`.
- To test face loading: `curl http://localhost:8000/debug/face-paths`.

## Key Files & Directories
- `src/majorSignL/api/main.py`: FastAPI app entrypoint
- `src/majorSignL/models/face_processor.py`: Face recognition logic
- `src/majorSignL/models/sign_classifier.py`: Sign language model
- `src/majorSignL/utils/mediapipe_processor.py`: MediaPipe integration
- `src/majorSignL/frontend/index.html`: Web interface

---
For more details, see `README.md` and code comments in each module.
