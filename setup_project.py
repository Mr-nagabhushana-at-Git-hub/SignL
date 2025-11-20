#!/usr/bin/env python3
"""
Project Synapse Setup Script
Automates the installation and configuration process
"""

import subprocess
import sys
from pathlib import Path
import os

def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"\n🔧 {description}")
    print(f"Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🎯 Project Synapse Setup Script")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    # Create necessary directories
    print("\n📁 Creating directory structure...")
    directories = [
        "src/data/known_faces",
        "src/data/models", 
        "data/training",
        "tests"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

    # Install the package in development mode
    if run_command("pip install -e .", "Installing project in development mode"):
        print("✅ Package installation complete")

    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Add face photos to src/data/known_faces/")
    print("2. Activate environment: mamba activate project-synapse")
    print("3. Start server: uvicorn src.majorSignL.api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("4. Open browser: http://localhost:8000/static/index.html")

if __name__ == "__main__":
    main()
