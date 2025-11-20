# SignL - Sign Language Recognition System
# PowerShell Startup Script

Write-Host "🚀 Starting SignL - Sign Language Recognition System" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Install/upgrade dependencies
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# Check CUDA availability
Write-Host "🔍 Checking CUDA availability..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Start the server
Write-Host "🌐 Starting FastAPI server..." -ForegroundColor Green
Write-Host "📍 Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "🎥 Web interface: http://localhost:8000/static/index.html" -ForegroundColor Cyan
Write-Host ""

python -m uvicorn signl.api.main:app --host 0.0.0.0 --port 8000 --reload
