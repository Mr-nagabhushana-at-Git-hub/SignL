# MajorSignL - Real-Time Sign Language Recognition with Face Recognition

A real-time AI system that combines **Sign Language Recognition** using MediaPipe and Transformer models with **Face Recognition** for person identification.

## 🚀 Features

- **Real-Time Sign Language Recognition**: Transformer-based model for recognizing sign language gestures
- **Face Recognition**: Identify people from pre-loaded face datasets organized by person folders
- **MediaPipe Integration**: Real-time pose, hand, and face landmark detection with smoothing filters
- **GPU Acceleration**: CUDA support for RTX 4060 and other NVIDIA GPUs
- **WebSocket Streaming**: Real-time video processing with web interface
- **REST API**: Complete API for managing faces and signs
- **Performance Monitoring**: FPS tracking and processing time metrics

## 📁 Project Structure

```
majorSignL/
├── src/
│   ├── data/
│   │   ├── fase_data/           # Person folders with face images
│   │   │   ├── Person Name 1/   # Folder named after person
│   │   │   ├── Person Name 2/   # Contains 10+ images per person
│   │   │   └── ...
│   │   ├── models/              # Pre-trained models
│   │   └── training/            # Sign language training data
│   └── majorSignL/
│       ├── api/                 # FastAPI server
│       ├── models/              # Face & sign processors
│       ├── utils/               # MediaPipe & filters
│       └── frontend/            # Web interface
├── env.yml                      # Conda environment (SignL)
├── start_server.sh             # Linux startup script
└── start_server.ps1            # Windows startup script
```

## 🛠️ Setup & Installation

### 1. Environment Setup (WSL2 + CUDA)

```bash
# Create conda environment
mamba env create -f env.yml

# Activate environment
mamba activate SignL

# Verify CUDA setup
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### 2. Face Recognition Setup

Your face data should be organized like this:
```
src/data/fase_data/
├── Aishwarya A/
│   ├── aish (1).jpg
│   ├── aish (2).jpg
│   └── ... (10+ images)
├── Chandra Shekara/
│   ├── chandra (1).jpg
│   └── ... (10+ images)
└── Nraju/
    ├── nagabhushana (1).jpg
    └── ... (10+ images)
```

The system will:
- Automatically load faces from person-named folders
- Create face encodings for each person using multiple images
- Cache encodings for faster startup
- Use face_recognition library with dlib

### 3. Sign Language Model Training

```bash
# Train the transformer model (if you have training data)
python src/majorSignL/train_model.py

# The model will be saved as:
# src/data/models/sign_language_transformer.pt
```

## 🔧 Advanced Features

### Face Recognition Models Integration
The system supports additional pre-trained face models:
- **FaceNet**: 177 embeddings loaded from `ds_model_facenet_detector_opencv_aligned_normalization_base_expand_0.pkl`
- **VGGFace**: 51 embeddings loaded from `ds_model_vggface_detector_opencv_aligned_normalization_base_expand_0.pkl`
- **VGG16**: Available but currently empty

These models provide additional verification for face recognition accuracy.

## 🚀 Running the Server

### Linux/WSL2:
```bash
chmod +x start_server.sh
./start_server.sh
```

### Windows:
```powershell
.\start_server.ps1
```

### Manual Start:
```bash
mamba activate SignL
cd src
python -m uvicorn majorSignL.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 API Endpoints

### Core Endpoints
- `GET /` - System status and capabilities
- `WebSocket /ws` - Real-time video processing
- `GET /static/index.html` - Web interface

### Face Recognition
- `GET /faces` - List known faces and info
- `POST /faces/refresh` - Refresh face cache
- `GET /debug/face-paths` - Debug face data paths

### Sign Language
- `GET /signs` - Sign classifier information  
- `POST /signs/reset` - Reset current sign sequence
- `POST /signs/confidence/{threshold}` - Set confidence threshold

## 🎯 Usage

1. **Start Server**: Run startup script
2. **Open Browser**: Go to `http://localhost:8000/static/index.html`
3. **Enable Camera**: Allow browser camera access
4. **Real-Time Processing**: 
   - Face recognition runs every 5th frame for performance
   - Sign recognition runs every 2nd frame
   - MediaPipe runs on every frame with filtering

## 🔧 Configuration

### Face Recognition Tuning
- **Threshold**: Adjust face matching threshold (default: 0.5)
- **Cache**: Face encodings are cached for faster startup
- **Performance**: Uses HOG model for speed, CNN for accuracy

### Sign Language Tuning  
- **Confidence**: Set prediction confidence threshold
- **Sequence Length**: 30 frames per sign (configurable)
- **Model**: Transformer architecture with attention

### Performance Optimization
- **GPU**: CUDA acceleration for PyTorch operations
- **Filtering**: One Euro Filter for landmark smoothing
- **Frame Skipping**: Different processing rates per component

## 🐛 Troubleshooting

### Face Recognition Issues
```bash
# Check if faces are loading
curl http://localhost:8000/debug/face-paths

# Refresh face cache
curl -X POST http://localhost:8000/faces/refresh

# Verify face_recognition installation
python -c "import face_recognition; print('OK')"
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version compatibility
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

### Performance Issues
- **Reduce frame rates**: Modify frame skipping intervals in main.py
- **Lower resolution**: Adjust frame resize in face_processor.py  
- **Disable features**: Comment out face/sign processing temporarily

## 📊 Performance Metrics

Expected performance on RTX 4060:
- **Overall FPS**: ~7-8 FPS
- **MediaPipe**: ~10-15ms per frame
- **Face Recognition**: ~50-100ms (every 5th frame)
- **Sign Classification**: ~20-40ms (every 2nd frame)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Test with your GPU/environment
4. Submit pull request

## 📝 License

- Proprarity Lisence 

---

**System Requirements:**
- Python 3.11
- NVIDIA GPU with CUDA support
- 8GB+ RAM
- Webcam for real-time processing

## Owner & Auther 
- Shri: Nagabhushana Raju S 

![alt text](image.png)