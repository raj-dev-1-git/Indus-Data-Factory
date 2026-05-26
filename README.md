# Audio Intelligence Pipeline & API

This repository contains a dual-mode Audio Intelligence system designed for multilingual speech-to-text (ASR) transcription and zero-shot temporal event detection.

1. **Offline Pipeline**: Batch processes audio scenes, performs language detection, ASR transcription, and semantic event detection.
2. **Online REST API**: A FastAPI backend that provides these capabilities as a real-time web service.

## Architecture

* **Perception Engine**: Utilizes `OpenAI Whisper` (large-v3) for multilingual Speech-to-Text, enhanced with native `initial_prompt` tuning for robust digit-to-word normalization. Audio event detection is powered by `LAION CLAP` operating in a zero-shot capacity.
* **Reasoning Engine**: Processes the transcriptions and events to provide deeper contextual insights.
* **Database**: SQLite with SQLAlchemy ORM to track inference results over time.

## Dataset & Scene Generation

The audio scenes evaluated in this pipeline are synthetically mixed from raw datasets:
* **Audio Events**: Sourced from Google's AudioSet ontology, specifically targeting threat signatures (e.g., gunfire, explosions, sirens) and contextual anchors (e.g., subway trains, dog barks). The raw clips are mined from YouTube using `yt-dlp`.
* **Speech**: Multilingual speech datasets (Marathi, Tamil, Bengali, Telugu).

To generate the evaluation audio scenes, the pipeline runs two key scripts:
1. `generate_mixer_logs.py`: Plans the temporal overlap and sequencing of speech and event audio.
2. `render_scenes.py`: Synthesizes the final `.wav` scenes using the generated logs.

## Project Structure

```text
├── app/                  # FastAPI Application
│   ├── services/         # Core AI Logic (Perception, Reasoning)
│   ├── static/           # Web Assets
│   ├── templates/        # HTML Templates
│   ├── database.py       # SQLite Configuration
│   ├── models.py         # SQLAlchemy Models
│   └── server.py         # Main FastAPI Endpoints
├── pipeline/             # Offline Batch Processing Scripts
│   ├── evaluate_asr.py   # Computes Word Error Rate (WER) and Character Error Rate (CER)
│   ├── run_perception.py # Batch ASR and Audio Event Detection
│   └── run_reasoning.py  # Batch Context Analysis
├── main.py               # Entry point to execute the offline pipeline
└── requirements.txt      # Python dependencies
```

## Setup Instructions

1. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   # source .venv/bin/activate # Linux/Mac
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Using Docker (Optional)**
   You can also run the web service via Docker:
   ```bash
   docker-compose up --build
   ```
   *Note: This assumes Ollama is running on your host machine.*

## How to Run

### 1. Run the Offline Pipeline
To process all audio scenes sequentially and generate metrics:
```bash
python main.py
```
This will run the audio mixing, ASR perception, reasoning, and evaluation in batch mode. The evaluation script outputs both Word Error Rate (WER) and Character Error Rate (CER), which is highly optimized for agglutinative Indic languages.

### 2. Run the Online Web Server
To start the real-time API:
```bash
fastapi dev app/server.py
```
* The web interface will be available at `http://127.0.0.1:8000`
* You can interact with the `/upload` endpoint to process `.wav` files dynamically.

## Recent Pipeline Improvements
* **F1 Label Mapping**: Re-aligned CLAP zero-shot text labels to map cleanly to the 5 base dataset event types (airplane, dog_bark, siren, traffic, crowd) via a canonical metadata field to accurately compute F1 scores.
* **CER Evaluation**: Integrated Character Error Rate calculation to accurately measure performance on highly agglutinative languages.
* **Whisper Normalization**: Integrated `whisper.normalizers` and native language `initial_prompt` conditioning to correctly parse numerical values into target language alphabets.
* **CLAP Temporal Tuning**: Adjusted sliding window detection logic (1.0s window, 0.5s stride) to aggressively filter false positives and correctly detect transient audio events. Both offline and online pipelines are now perfectly synced.
