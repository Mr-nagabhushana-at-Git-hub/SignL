#!/bin/bash
# run_server_direct.sh - Start server when already in activated environment

echo "🚀 Starting MajorSignL Server (Direct Mode)..."
echo "=========================================="

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check if we're in the right environment
python -c "import sys; print(f'Using Python: {sys.executable}')"

# Quick dependency check
echo "🔍 Checking dependencies..."
python -c "
import sys
missing = []
try:
    import torch
    print('✅ PyTorch available')
except ImportError:
    missing.append('torch')

try:
    import fastapi, uvicorn
    print('✅ FastAPI/Uvicorn available')
except ImportError:
    missing.append('fastapi/uvicorn')

try:
    import mediapipe
    print('✅ MediaPipe available')
except ImportError:
    missing.append('mediapipe')

try:
    import cv2
    print('✅ OpenCV available')
except ImportError:
    missing.append('opencv')

try:
    import face_recognition
    print('✅ Face Recognition available')
except ImportError:
    missing.append('face_recognition')

if missing:
    print(f'❌ Missing: {missing}')
    print('Make sure you are in the SignL environment!')
    sys.exit(1)
else:
    print('✅ All core dependencies found')
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed. Make sure you're in the SignL environment:"
    echo "mamba activate SignL"
    exit 1
fi

echo "🎯 Checking face data..."
if [ -d "src/data/fase_data" ]; then
    face_count=$(find src/data/fase_data -maxdepth 1 -type d | wc -l)
    echo "📁 Found $((face_count-1)) person folders in fase_data"
else
    echo "⚠️ Face data directory not found at src/data/fase_data"
fi

echo "🌐 Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:8000/static/index.html"
echo "API docs at: http://localhost:8000/docs"
echo "=========================================="

# Start the server
cd src
python -m uvicorn majorSignL.api.main:app --host 0.0.0.0 --port 8000 --reload
