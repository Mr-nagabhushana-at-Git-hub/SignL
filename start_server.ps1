# start_server.ps1 - Windows PowerShell version

Write-Host "🚀 Starting MajorSignL Server..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

# Check if in correct directory
if (!(Test-Path "pyproject.toml")) {
    Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

Write-Host "📦 Activating conda environment 'SignL'..." -ForegroundColor Yellow
conda activate SignL

Write-Host "🔍 Checking Python version and CUDA..." -ForegroundColor Blue
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')" 2>$null

Write-Host "🎯 Checking face data..." -ForegroundColor Magenta
if (Test-Path "src\data\fase_data") {
    $faceCount = (Get-ChildItem -Path "src\data\fase_data" -Directory).Count
    Write-Host "📁 Found $faceCount person folders in fase_data" -ForegroundColor Green
} else {
    Write-Host "⚠️ Face data directory not found at src\data\fase_data" -ForegroundColor Yellow
}

Write-Host "🤟 Checking sign language model..." -ForegroundColor Blue
if (Test-Path "src\data\models\sign_language_transformer.pt") {
    Write-Host "✅ Sign language PyTorch model found" -ForegroundColor Green
} elseif (Test-Path "src\data\models\sign_language_transformer.h5") {
    Write-Host "⚠️ Found Keras model (.h5) but expecting PyTorch (.pt)" -ForegroundColor Yellow
    Write-Host "Please retrain with: python src\majorSignL\train_model.py" -ForegroundColor Yellow
} else {
    Write-Host "⚠️ Sign language model not found. Running in demo mode." -ForegroundColor Yellow
    Write-Host "To train: python src\majorSignL\train_model.py" -ForegroundColor Yellow
}

Write-Host "🌐 Starting FastAPI server..." -ForegroundColor Green
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend will be available at: http://localhost:8000/static/index.html" -ForegroundColor Cyan
Write-Host "API docs at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Start the server
Set-Location src
python -m uvicorn majorSignL.api.main:app --host 0.0.0.0 --port 8000 --reload
