"""
Unified Perception Service — wraps the same logic as pipeline/run_perception.py.
Uses Faster-Whisper (large-v3-turbo, int8, CPU) and PANNs CNN14 with
multi-scale sliding windows for audio event detection.
"""

import os
from pathlib import Path

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from panns_inference import AudioTagging, labels as AUDIOSET_LABELS

BASE = Path(__file__).resolve().parent.parent.parent

# ── AudioSet class mapping ───────────────────────────────────────
# Map our event keys to one or more AudioSet label names.
# PANNs CNN14 outputs a probability for each of the 527
# AudioSet classes; we take the max across mapped labels.

EVENT_TO_AUDIOSET = {
    "civil_defense_siren": ["Civil defense siren", "Siren"],
    "dog_bark": ["Bark", "Dog"],
    "gunfire": ["Gunshot, gunfire"],
    "subway_train": ["Subway, metro, underground"],
}

EVENT_KEYS = list(EVENT_TO_AUDIOSET.keys())

# Build index lookup: event_key -> list of AudioSet class indices
_label_to_idx = {lab: i for i, lab in enumerate(AUDIOSET_LABELS)}

EVENT_INDICES = {}
for _key, _audioset_names in EVENT_TO_AUDIOSET.items():
    _indices = []
    for _name in _audioset_names:
        if _name in _label_to_idx:
            _indices.append(_label_to_idx[_name])
        else:
            print(f"WARNING: AudioSet label '{_name}' not found for '{_key}'")
    EVENT_INDICES[_key] = _indices

LANG_MAP = {
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
}

WINDOW_CONFIGS = [
    {"seconds": 1.0, "stride": 0.5},
    {"seconds": 2.0, "stride": 1.0},
    {"seconds": 4.0, "stride": 2.0},
]

EVENT_THRESHOLDS = {
    "civil_defense_siren": 0.08,
    "dog_bark": 0.08,
    "gunfire": 0.05,
    "subway_train": 0.01,
}

INITIAL_PROMPTS = {
    "mr": "कृपया सर्व आकडे शब्दांत लिहा.",
    "ta": "தயவுசெய்து அனைத்து எண்களையும் வார்த்தைகளில் எழுதவும்.",
    "te": "దయచేసి అన్ని సంఖ్యలను పదాలలో రాయండి.",
}

# ── Load models once at startup ─────────────────────────────────────

print("Loading Faster-Whisper (large-v3-turbo, int8, CPU)...")
whisper_model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
)

print("Loading PANNs CNN14...")
panns_model = AudioTagging(checkpoint_path=None, device="cpu")


def run_perception(audio_path, language_hint=None):
    """
    Run ASR + event detection on an audio file.
    Returns the same perception dict as pipeline/run_perception.py.
    """
    lang_code = None
    if language_hint:
        lang_code = LANG_MAP.get(language_hint, language_hint)

    # ── ASR ──────────────────────────────────────────────────────
    try:
        audio_for_whisper, _ = librosa.load(audio_path, sr=16000, mono=True)

        whisper_kwargs = dict(
            beam_size=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            repetition_penalty=1.2,
        )

        if lang_code:
            whisper_kwargs["language"] = lang_code
            whisper_kwargs["initial_prompt"] = INITIAL_PROMPTS.get(lang_code, "")

        segments, info = whisper_model.transcribe(
            audio_for_whisper, **whisper_kwargs
        )
        segments = list(segments)
        transcript = " ".join(seg.text for seg in segments)
        detected_lang = lang_code or info.language

    except Exception as e:
        print(f"ASR failed: {e}")
        return {
            "detected_language": "unknown",
            "full_transcript": "Transcription failed.",
            "speech": [],
            "events": [],
        }

    speech_segments = [
        {
            "type": "speech",
            "content": seg.text.strip(),
            "time": f"{seg.start:.2f}-{seg.end:.2f}",
            "confidence": round(float(seg.avg_logprob), 3),
        }
        for seg in segments
    ]

    # ── PANNs Event Detection ───────────────────────────────────
    # PANNs expects 32 kHz audio
    audio_32k, sr = librosa.load(audio_path, sr=32000, mono=True)
    total_length = len(audio_32k)

    best_score_per_event = {k: -999.0 for k in EVENT_KEYS}
    best_window_per_event = {k: "" for k in EVENT_KEYS}

    def _score_chunk(chunk, t_start, t_end):
        """Compute per-event max probability for one chunk."""
        if np.max(np.abs(chunk)) < 0.01:
            return

        # PANNs expects (batch, samples)
        waveform = chunk[np.newaxis, :]
        (clipwise_output, _) = panns_model.inference(waveform)
        probs = clipwise_output[0]  # shape (527,)

        for key in EVENT_KEYS:
            max_prob = max(
                float(probs[idx])
                for idx in EVENT_INDICES[key]
            )
            if max_prob > best_score_per_event[key]:
                best_score_per_event[key] = max_prob
                best_window_per_event[key] = f"{t_start:.2f}-{t_end:.2f}"

    # Multi-scale sliding windows
    for cfg in WINDOW_CONFIGS:
        win = int(cfg["seconds"] * sr)
        stride = int(cfg["stride"] * sr)
        for start_idx in range(0, total_length - win, stride):
            end_idx = start_idx + win
            _score_chunk(
                audio_32k[start_idx:end_idx],
                start_idx / sr,
                end_idx / sr,
            )

    # Full-clip pass
    _score_chunk(audio_32k, 0.0, total_length / sr)

    # Collect detected events using per-event thresholds
    detected = []
    for key in EVENT_KEYS:
        sc = best_score_per_event[key]
        if sc >= EVENT_THRESHOLDS[key]:
            detected.append(
                {
                    "type": "sound_event",
                    "content": key.replace("_", " "),
                    "time": best_window_per_event[key],
                    "confidence": round(sc, 3),
                }
            )

    # Dynamic filtering: keep events within 0.20 of top confidence
    detected.sort(key=lambda x: x["confidence"], reverse=True)
    if detected:
        top_conf = detected[0]["confidence"]
        sound_events = [d for d in detected if d["confidence"] >= (top_conf - 0.20)]
    else:
        sound_events = []

    return {
        "detected_language": detected_lang,
        "full_transcript": transcript,
        "speech": speech_segments,
        "events": sound_events,
    }
