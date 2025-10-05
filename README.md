# OpenAI-Compatible TTS Server with Piper + RVC

A FastAPI server that provides OpenAI-compatible text-to-speech endpoints using:
- **Piper TTS** for high-quality neural text-to-speech
- **RVC (Retrieval-based Voice Conversion)** to convert to Hatsune Miku voice
- **FFmpeg** for audio format conversion

## Features

- 🎯 **OpenAI-compatible API** - Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- 🎤 **High-quality TTS** - Uses Piper neural TTS for natural-sounding speech
- 🎵 **Voice conversion** - RVC conversion to Hatsune Miku voice character
- 🔄 **Multiple formats** - Supports MP3, WAV, FLAC, AAC, Opus, PCM
- ⚡ **Async processing** - FastAPI with async audio processing
- 📊 **Automatic setup** - Downloads models automatically on first run

## Quick Start

### Linux/macOS
```bash
# Run the automated setup
./setup.sh

# Start the server
source venv/bin/activate
python tts_server.py
```

### Windows
```batch
# Run the automated setup
setup.bat

# Start the server
venv\Scripts\activate.bat
python tts_server.py
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for audio conversion)
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg  
# Windows: Download from https://ffmpeg.org

# Start server
python tts_server.py
```

## Usage

The server runs on `http://localhost:8000` and provides OpenAI-compatible endpoints:

### Generate Speech
```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello world! This will be spoken in Hatsune Miku voice.",
    "voice": "alloy",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

### Python Example
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello from Hatsune Miku!",
        "voice": "alloy",
        "response_format": "mp3"
    }
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | "tts-1" | TTS model (currently ignored) |
| `input` | string | *required* | Text to convert (max 4096 chars) |
| `voice` | string | "alloy" | Voice name (mapped internally) |
| `response_format` | string | "mp3" | Output format (mp3, wav, flac, etc.) |
| `speed` | float | 1.0 | Speech speed (0.25 - 4.0) |

## Testing

Test the server with the included test script:

```bash
# Make sure server is running, then:
python test_server.py
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Models

The server automatically downloads required models on first run:

- **Piper TTS Model**: `en_US-amy-medium` (English, female voice)
- **RVC Model**: Hatsune Miku voice conversion model

Models are cached locally in the `models/` directory.

## Architecture

1. **Input**: Text via OpenAI-compatible API
2. **Piper TTS**: Generates initial speech audio (WAV)
3. **RVC Conversion**: Converts voice to Hatsune Miku character
4. **Format Conversion**: Uses FFmpeg to convert to requested format
5. **Output**: Audio file in requested format

## Requirements

- Python 3.8+
- FFmpeg (for audio conversion)
- ~2GB disk space for models
- 4GB+ RAM recommended

## Troubleshooting

- **FFmpeg not found**: Install FFmpeg and ensure it's in your PATH
- **Model download fails**: Check internet connection and disk space
- **RVC conversion slow**: RVC processing is CPU-intensive; consider using GPU
- **Memory issues**: Reduce concurrent requests or upgrade RAM

## License

This project is open source. Model licenses:
- Piper TTS: MIT License
- RVC models: Check individual model licenses
