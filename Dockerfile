# Dockerfile for OpenAI-Compatible TTS Server with Autotune
FROM python:3.11-slim

# Install system dependencies (including audio processing libs)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    build-essential \
    libsndfile1-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS binary
RUN wget -O piper.tar.gz "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz" \
    && tar -xzf piper.tar.gz \
    && mv piper/piper /usr/local/bin/ \
    && rm -rf piper piper.tar.gz

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install patched fairseq for Python 3.11+ compatibility
RUN pip install --no-cache-dir https://github.com/One-sixth/fairseq/archive/main.zip

# Copy application code
COPY tts_server.py .

# Create models directory
RUN mkdir -p models/piper models/rvc

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "tts_server_autotuned.py"]
