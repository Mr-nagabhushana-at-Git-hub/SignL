#!/bin/bash
# Quick development server start (assumes dependencies are installed)

echo "🚀 Quick Start - SignL Server"
cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start server
python3 -m uvicorn signl.api.main:app --host 0.0.0.0 --port 8000 --reload
