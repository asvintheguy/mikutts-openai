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

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install patched fairseq for Python 3.11+ compatibility
RUN pip install --no-cache-dir https://github.com/One-sixth/fairseq/archive/main.zip

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "tts_server.py"]
