# SignL Setup Guide

## First-Time Setup Checklist

### 1. ✅ Install System Dependencies

For Ubuntu/Debian (Codespaces):
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    libopencv-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev
```

### 2. ✅ Create Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3. ✅ Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This may take 10-15 minutes depending on your connection.

### 4. ✅ Prepare Face Data

1. Create person folders in `signl/data/face_data/`:
   ```bash
   mkdir -p signl/data/face_data/Person_Name
   ```

2. Add 10-20 clear photos of each person's face

3. Supported formats: JPG, JPEG, PNG

4. File naming doesn't matter - folder name is used as the person's name

### 5. ✅ Optional: Add Pre-trained Sign Language Model

If you have a trained model:
```bash
cp your_model.pt signl/data/models/sign_language_transformer.pt
```

If not, the system will run in demo mode.

### 6. ✅ Start the Server

```bash
./start.sh  # Linux/Mac/Codespaces
# OR
.\start.ps1  # Windows
```

### 7. ✅ Access the Interface

Open in your browser:
- http://localhost:8000/static/index.html

## Troubleshooting

### CUDA Not Available
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Face Recognition Issues
```bash
# Verify face_recognition installation
python -c "import face_recognition; print('OK')"

# Reinstall dlib if needed
pip install --upgrade dlib face-recognition
```

### MediaPipe Issues
```bash
# Reinstall MediaPipe
pip install --upgrade mediapipe opencv-python
```

### Port Already in Use
```bash
# Use a different port
python -m uvicorn signl.api.main:app --host 0.0.0.0 --port 8080 --reload
```

## Performance Optimization

### For CPU-Only Systems
- Face recognition will use HOG model (faster)
- Sign language classification will be slower
- Consider reducing frame processing intervals

### For GPU Systems
- Ensure CUDA is properly installed
- PyTorch should detect GPU automatically
- Check with: `nvidia-smi`

## Development Tips

### Quick Restart
```bash
./dev.sh  # Faster startup, assumes dependencies installed
```

### Add New Person
1. Create folder: `signl/data/face_data/New_Person/`
2. Add photos
3. Restart server OR call API: `POST /faces/refresh`

### View Logs
The server outputs detailed logs including:
- Face recognition results
- Sign language predictions
- Emotion detection
- Performance metrics

## Next Steps

1. ✅ Add your face images
2. ✅ Test face recognition
3. ✅ Test sign language gestures
4. ✅ Explore API endpoints at `/docs`
5. ✅ Fine-tune emotion detection
6. ✅ Train custom sign language model (optional)
