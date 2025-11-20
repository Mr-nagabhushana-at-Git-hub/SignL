#!/bin/bash
# start_server.sh - Start the MajorSignL server

echo "🚀 Starting MajorSignL Server..."
echo "=========================================="

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Determine environment manager
if command -v mamba &> /dev/null; then
    ENV_MANAGER="mamba"
elif command -v conda &> /dev/null; then
    ENV_MANAGER="conda"
else
    echo "❌ Neither conda nor mamba found!"
    exit 1
fi

# Check if conda environment exists
if ! $ENV_MANAGER env list | grep -q "SignL"; then
    echo "❌ Conda environment 'SignL' not found."
    echo "Please create it first with:"
    echo "$ENV_MANAGER env create -f env.yml"
    exit 1
fi

echo "� Checking Python version and CUDA using SignL environment..."
$ENV_MANAGER run -n SignL python -c "import sys; print(f'Python: {sys.version}')"
$ENV_MANAGER run -n SignL python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || echo "⚠️ PyTorch not available in SignL environment"

echo "🎯 Checking face data..."
if [ -d "src/data/fase_data" ]; then
    face_count=$(find src/data/fase_data -maxdepth 1 -type d | wc -l)
    echo "📁 Found $((face_count-1)) person folders in fase_data"
else
    echo "⚠️ Face data directory not found at src/data/fase_data"
fi

echo "🤟 Checking dependencies..."
$ENV_MANAGER run -n SignL python -c "
try:
    import fastapi, uvicorn, mediapipe, cv2, face_recognition
    print('✅ Core dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
"

echo "🤟 Checking sign language model..."
if [ -f "src/data/models/sign_language_transformer.pt" ]; then
    echo "✅ Sign language model found"
else
    echo "⚠️ Sign language model not found. You may need to train it first."
    echo "Run: $ENV_MANAGER run -n SignL python src/majorSignL/train_model.py"
fi

echo "🌐 Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:8000/static/index.html"
echo "API docs at: http://localhost:8000/docs"
echo "=========================================="

# Start the server using mamba/conda run
cd src
$ENV_MANAGER run -n SignL python -m uvicorn majorSignL.api.main:app --host 0.0.0.0 --port 8000 --reload
