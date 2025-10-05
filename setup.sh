#!/bin/bash
# Setup script for OpenAI-compatible TTS server with Piper + RVC

set -e

echo "🚀 Setting up OpenAI-compatible TTS server with Piper + RVC..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or later and try again."
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Installing..."

    # Detect OS and install ffmpeg
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt &> /dev/null; then
            sudo apt update && sudo apt install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S ffmpeg
        else
            echo "❌ Please install ffmpeg manually for your Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "❌ Please install Homebrew and then run: brew install ffmpeg"
            exit 1
        fi
    else
        echo "❌ Please install ffmpeg manually for your operating system"
        echo "Visit: https://ffmpeg.org/download.html"
        exit 1
    fi
fi

echo "✅ FFmpeg is available"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the server:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the server: python tts_server.py"
echo ""
echo "The server will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Example usage:"
echo 'curl -X POST "http://localhost:8000/v1/audio/speech" \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"model": "tts-1", "input": "Hello world!", "voice": "alloy"}'"'"' \'
echo '  --output speech.mp3'
