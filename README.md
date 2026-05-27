# Audio Intelligence Pipeline & API

This repository contains a dual-mode Audio Intelligence system designed for multilingual Indic speech-to-text (ASR) transcription and zero-shot temporal event detection.

1. **Offline Pipeline**: Batch processes audio scenes, performs language detection, ASR transcription, and semantic event detection.
2. **Online REST API**: A FastAPI backend that provides these capabilities as a real-time web service.

## Architecture

* **Perception Engine**: Utilizes `Faster-Whisper` (large-v3-turbo, int8 quantized for CPU) for multilingual Speech-to-Text, enhanced with native `initial_prompt` tuning for robust digit-to-word normalization. Audio event detection is powered by `PANNs CNN14` (Pre-trained Audio Neural Networks) operating as a supervised AudioSet classifier with multi-scale sliding windows (1s/0.5s stride + 2s/1.0s stride + 4s/2.0s stride + full-clip) for temporal localization.
* **Reasoning Engine**: Processes the transcriptions and events via a local LLM (Qwen 2.5 1.5B through Ollama) to provide contextual insights.
* **Database**: SQLite with SQLAlchemy ORM to track inference results over time.

## Supported Languages

| Language | Code | Status |
|----------|------|--------|
| Marathi  | mr   | ✅ Active |
| Tamil    | ta   | ✅ Active |
| Telugu   | te   | ✅ Active |

## Event Detection Labels

The PANNs CNN14 detector targets 4 threat-relevant and contextual event categories (mapped to AudioSet classes):

`civil_defense_siren` · `dog_bark` · `gunfire` · `subway_train`

## Dataset & Scene Generation

The audio scenes evaluated in this pipeline are synthetically mixed from raw datasets:
* **Audio Events**: Sourced from Google's AudioSet ontology, specifically targeting threat signatures (e.g., gunfire, explosions, sirens) and contextual anchors (e.g., subway trains, dog barks). The raw clips are mined from YouTube using `yt-dlp`.
* **Speech**: Multilingual speech datasets (Marathi, Tamil, Telugu).

To generate the evaluation audio scenes, the pipeline runs two key scripts:
1. `generate_mixer_logs.py`: Plans the temporal overlap and sequencing of speech and event audio.
2. `render_scenes.py`: Synthesizes the final `.wav` scenes using the generated logs.

## Project Structure

```text
├── app/                        # FastAPI Application
│   ├── services/               # Core AI Logic (Perception, Reasoning)
│   │   ├── perception_service.py   # Unified perception engine (same as pipeline)
│   │   └── reasoning_service.py    # LLM reasoning via Ollama
│   ├── static/                 # Web Assets (JS, CSS)
│   ├── templates/              # HTML Templates
│   ├── database.py             # SQLite Configuration
│   ├── models.py               # SQLAlchemy Models
│   └── server.py               # Main FastAPI Endpoints
├── pipeline/                   # Offline Batch Processing Scripts
│   ├── generate_mixer_logs.py  # Synthetic scene metadata generator
│   ├── render_scenes.py        # Audio scene renderer/mixer
│   ├── run_perception.py       # Batch ASR + PANNs event detection
│   ├── evaluate_asr.py         # Computes WER, CER, and Event F1
│   └── run_reasoning.py        # Batch LLM context analysis
├── main.py                     # Entry point for the offline pipeline
└── requirements.txt            # Python dependencies
```

## Model Downloads

The PANNs CNN14 checkpoint (~320 MB) is automatically downloaded on first run via `panns_inference`. No manual model placement is required.

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

3. **Install Ollama** (for reasoning layer)
   - Download from [ollama.com](https://ollama.com)
   - Pull the model: `ollama pull qwen2.5:1.5b`
   - Optionally set `OLLAMA_EXE` environment variable if Ollama is not in PATH

## How to Run

### 1. Run the Offline Pipeline
To process all audio scenes sequentially and generate metrics:
```bash
python main.py
```
This will run the audio mixing, ASR perception, reasoning, and evaluation in batch mode. The evaluation script outputs Word Error Rate (WER), Character Error Rate (CER), and Event Detection F1.

### 2. Run the Online Web Server
To start the real-time API:
```bash
fastapi dev app/server.py
```
* The web interface will be available at `http://127.0.0.1:8000`
* You can interact with the `/upload` endpoint to process `.wav` files dynamically.

## Key Design Decisions

* **CPU-Only Inference**: All models run on CPU with int8 quantization. Whisper uses `large-v3-turbo` (optimized for speed) with beam search and temperature fallbacks.
* **PANNs CNN14 Classification**: Each event type maps to one or more AudioSet classes (527 total). The max sigmoid probability across mapped classes is used per event, providing calibrated confidence scores without prompt engineering.
* **Multi-Scale Windowing**: Short windows (1s) catch impulsive events (gunfire), medium windows (2s/4s) catch sustained events (sirens), and full-clip gives PANNs broader context.
* **Unified Codepath**: Both the offline pipeline and web API use the exact same perception and reasoning engines — same models, same parameters, same results.
* **CER over WER**: Character Error Rate is the primary metric for agglutinative Indic languages where word boundary differences inflate WER artificially.
