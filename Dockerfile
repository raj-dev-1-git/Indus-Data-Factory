FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for PyAudio, soundfile, librosa, etc.
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# The app expects the database in the 'database' folder at the project root
# and uploads in the 'uploads' folder. These should be mounted as volumes if persistence is needed.

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
