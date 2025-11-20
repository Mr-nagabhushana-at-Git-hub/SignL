#!/bin/bash
# SignL Startup Script
# Starts the SignL application server

set -e

echo "🚀 Starting SignL - Sign Language Recognition System"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || echo "PyTorch not installed yet"

# Start the server
echo "🌐 Starting FastAPI server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "🎥 Web interface: http://localhost:8000/static/index.html"
echo ""

cd "$(dirname "$0")"
python3 -m uvicorn signl.api.main:app --host 0.0.0.0 --port 8000 --reload
