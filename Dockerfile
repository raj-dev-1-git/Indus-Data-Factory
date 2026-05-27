FROM python:3.11-slim AS base

# System dependencies required by librosa, soundfile, and ffmpeg-python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ app/
COPY main.py .
COPY pipeline/ pipeline/
COPY archive/prompts/ archive/prompts/

# Create runtime directories
RUN mkdir -p uploads database scenes/audio scenes/metadata perception reasoning

# Non-root user
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default: run the web server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
