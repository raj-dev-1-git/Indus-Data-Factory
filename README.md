# Indus Data Factory

A dual-mode audio intelligence system for multilingual Indic speech-to-text transcription and zero-shot temporal event detection. Operates as both an offline batch pipeline and a real-time REST API.

## Architecture

The system is composed of three layers:

**Perception Engine** -- Handles raw audio analysis through two parallel subsystems:
- *Speech-to-Text*: Faster-Whisper (large-v3, int8 quantized for CPU) with beam search, temperature fallbacks, and VAD filtering. Language-specific initial prompts enforce digit-to-word normalization for Indic scripts.
- *Event Detection*: PANNs CNN14 (Pre-trained Audio Neural Networks) operating as a supervised AudioSet classifier. Multi-scale sliding windows (1s/0.5s stride, 2s/1.0s stride, 4s/2.0s stride, plus full-clip) provide temporal localization across impulsive and sustained event types.

**Reasoning Engine** -- Feeds structured perception output to a local LLM (Qwen 2.5 1.5B via Ollama) for contextual scene analysis, environment inference, and risk assessment.

**Database** -- SQLite with SQLAlchemy ORM for persistent storage of inference results.

### Supported Languages

| Language | Code |
|----------|------|
| Marathi  | mr   |
| Tamil    | ta   |
| Telugu   | te   |

### Event Detection Targets

The detector covers four event categories mapped to AudioSet classes:

| Event Key              | AudioSet Classes                    |
|------------------------|-------------------------------------|
| `civil_defense_siren`  | Civil defense siren, Siren          |
| `dog_bark`             | Bark, Dog                           |
| `gunfire`              | Gunshot, gunfire                    |
| `subway_train`         | Subway, metro, underground          |

## Prerequisites

- Python 3.10 or later
- [FFmpeg](https://ffmpeg.org/) installed and available on PATH
- [Ollama](https://ollama.com/) installed (for the reasoning layer)
- Approximately 4 GB of disk space for model weights (downloaded automatically on first run)

## Installation

```bash
# Clone the repository
git clone https://github.com/raj-dev-1-git/Indus-Data-Factory.git
cd Indus-Data-Factory

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source .venv/bin/activate       # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Pull the reasoning model
ollama pull qwen2.5:1.5b
```

Copy `.env.example` to `.env` and adjust values if needed:

```bash
cp .env.example .env
```

## Usage

### Offline Pipeline

Processes all audio scenes sequentially through mixing, perception, reasoning, and evaluation:

```bash
python main.py
```

The evaluation stage reports Word Error Rate (WER), Character Error Rate (CER), and Event Detection F1 score.

### Web Server

Start the FastAPI application for real-time inference:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

The interface is available at `http://localhost:8000`. Upload `.wav` files through the web UI or POST directly to the `/upload` endpoint.

### Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

This starts both the web server and an Ollama instance. The server is accessible at `http://localhost:8000`.

To run the offline pipeline as a one-shot job:

```bash
docker compose run --rm pipeline
```

## Configuration

All runtime parameters are configurable through environment variables. See `.env.example` for the full list:

| Variable               | Default          | Description                          |
|------------------------|------------------|--------------------------------------|
| `OLLAMA_EXE`           | `ollama`         | Path to the Ollama binary            |
| `OLLAMA_MODEL`         | `qwen2.5:1.5b`  | LLM model identifier                 |
| `WHISPER_MODEL`        | `large-v3`       | Whisper model size                   |
| `WHISPER_DEVICE`       | `cpu`            | Inference device (cpu or cuda)       |
| `WHISPER_COMPUTE_TYPE` | `int8`           | Quantization type                    |
| `PANNS_DEVICE`         | `cpu`            | PANNs inference device               |
| `APP_HOST`             | `0.0.0.0`        | Server bind address                  |
| `APP_PORT`             | `8000`           | Server port                          |

## Project Structure

```
app/
    server.py                   FastAPI application and endpoints
    database.py                 SQLite engine and session configuration
    models.py                   SQLAlchemy ORM models
    services/
        perception_service.py   Unified perception engine (ASR + event detection)
        reasoning_service.py    LLM reasoning via Ollama subprocess
    static/                     Frontend assets (JavaScript, CSS)
    templates/                  Jinja2 HTML templates

pipeline/
    generate_mixer_logs.py      Synthetic scene metadata generator
    render_scenes.py            Audio scene renderer and mixer
    run_perception.py           Batch ASR and event detection
    evaluate_asr.py             WER, CER, and Event F1 evaluation
    run_reasoning.py            Batch LLM reasoning

archive/
    prompts/
        prompt_v1.txt           System prompt for the reasoning engine

main.py                         Entry point for the offline pipeline
requirements.txt                Python dependencies
Dockerfile                      Container build definition
docker-compose.yml              Multi-service orchestration
.env.example                    Environment variable reference
```

## Dataset and Scene Generation

Audio scenes are synthetically mixed from two source categories:

- **Speech**: Multilingual speech corpora in Marathi, Tamil, and Telugu, stored under `clean/` with corresponding transcript files.
- **Events**: Audio clips sourced from Google AudioSet, organized by event type under `dataset/events/`.

The pipeline generates scenes through two stages: `generate_mixer_logs.py` creates temporal layout metadata, and `render_scenes.py` mixes the final waveforms.

## Evaluation Metrics

- **CER (Character Error Rate)**: Primary metric for Indic languages. Character-level edit distance avoids the inflated error rates that WER produces on agglutinative scripts with inconsistent word boundaries.
- **WER (Word Error Rate)**: Secondary metric, reported alongside CER for reference.
- **Event Detection F1**: Set-level precision, recall, and F1 computed over detected vs. ground truth event labels per scene.

## Design Rationale

- **CPU-only inference**: All models run on CPU with int8 quantization. No GPU required for deployment.
- **PANNs over zero-shot**: Supervised AudioSet classification with explicit class mapping provides calibrated confidence scores without prompt engineering overhead.
- **Multi-scale windowing**: Short windows (1s) capture impulsive events such as gunfire. Medium windows (2s, 4s) capture sustained events such as sirens. Full-clip inference provides global context.
- **Unified codepath**: The offline pipeline and web API share the same perception and reasoning modules, same models, same parameters.
- **CER as primary metric**: Word boundary differences across Indic language tokenizers inflate WER artificially. CER provides a more stable and comparable measurement.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
