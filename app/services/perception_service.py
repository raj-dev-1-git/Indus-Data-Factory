"""
Unified Perception Service — wraps the same logic as pipeline/run_perception.py.
Uses Faster-Whisper (large-v3-turbo, int8, CPU) and LAION CLAP with
multi-prompt ensemble + multi-scale sliding windows.
"""

import os
from pathlib import Path

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
import laion_clap

BASE = Path(__file__).resolve().parent.parent.parent

# ── Multi-prompt ensemble: 3 prompts per event ──────────────────────
EVENT_PROMPTS = {
    "car_honk": [
        "a car horn honking on the road",
        "vehicle horn, car horn, honking",
        "loud beeping car horn",
    ],
    "civil_defense_siren": [
        "a civil defense siren or air raid siren wailing",
        "emergency warning siren, loud alarm siren",
        "civil defense siren",
    ],
    "dog_bark": [
        "a dog barking loudly",
        "dog bark, barking dog",
        "aggressive dog barking and growling",
    ],
    "explosion": [
        "a loud explosion or blast",
        "bomb explosion, detonation, loud bang",
        "explosion",
    ],
    "fighter_jet_engine": [
        "jet engine roaring, aircraft engine noise",
        "fighter jet flying overhead",
        "loud turbine engine, jet engine",
    ],
    "gunfire": [
        "the sound of gunfire or gunshots",
        "gunshot, gunfire, shooting",
        "firearms discharge, gun blast",
    ],
    "subway_train": [
        "a subway train or metro train passing",
        "train on tracks, railway, rumbling train",
        "subway, metro, underground train",
    ],
}

EVENT_KEYS = list(EVENT_PROMPTS.keys())

# Flatten all prompts; remember which event each prompt belongs to
_ALL_PROMPTS = []
_PROMPT_TO_EVENT_IDX = []
for _i, _key in enumerate(EVENT_KEYS):
    for _prompt in EVENT_PROMPTS[_key]:
        _ALL_PROMPTS.append(_prompt)
        _PROMPT_TO_EVENT_IDX.append(_i)

LANG_MAP = {
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
}

# Multi-scale windows: short (3 s) catches impulsive events,
# long (5 s) catches sustained events, full-clip gives CLAP its
# native 10 s context.
WINDOW_CONFIGS = [
    {"seconds": 3, "stride": 1.0},
    {"seconds": 5, "stride": 1.5},
]
EVENT_THRESHOLD = 0.20

INITIAL_PROMPTS = {
    "mr": "कृपया सर्व आकडे शब्दांत लिहा.",
    "ta": "தயவுசெய்து அனைத்து எண்களையும் வார்த்தைகளில் எழுதவும்.",
    "te": "దయచేసి అన్ని సంఖ్యలను పదాలలో రాయండి.",
}

# ── Load models once at startup ─────────────────────────────────────

print("Loading Faster-Whisper (large-v3-turbo, int8, CPU)...")
whisper_model = WhisperModel(
    "large-v3-turbo",
    device="cpu",
    compute_type="int8",
)

print("Loading CLAP (HTSAT-base)...")
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
clap_model.load_ckpt(
    ckpt=str(BASE / "music_speech_audioset_epoch_15_esc_89.98.pt")
)

# Pre-compute text embeddings for all prompts
_text_embed = clap_model.get_text_embedding(_ALL_PROMPTS, use_tensor=False)
_text_embed = _text_embed / np.linalg.norm(_text_embed, axis=-1, keepdims=True)


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

    # ── CLAP Event Detection ────────────────────────────────────
    audio_48k, sr = librosa.load(audio_path, sr=48000, mono=True)
    total_length = len(audio_48k)

    best_score_per_event = {k: -999.0 for k in EVENT_KEYS}
    best_window_per_event = {k: "" for k in EVENT_KEYS}

    def _score_chunk(chunk, t_start, t_end):
        """Compute per-event max-over-prompts cosine sim for one chunk."""
        if np.max(np.abs(chunk)) < 0.01:
            return
        audio_embed = clap_model.get_audio_embedding_from_data(
            x=[chunk], use_tensor=False
        )
        audio_embed = audio_embed / np.linalg.norm(
            audio_embed, axis=-1, keepdims=True
        )
        sim = (audio_embed @ _text_embed.T)[0]

        for ei, key in enumerate(EVENT_KEYS):
            prompt_scores = [
                float(sim[pi])
                for pi, eidx in enumerate(_PROMPT_TO_EVENT_IDX)
                if eidx == ei
            ]
            max_s = max(prompt_scores)
            if max_s > best_score_per_event[key]:
                best_score_per_event[key] = max_s
                best_window_per_event[key] = f"{t_start:.2f}-{t_end:.2f}"

    # Multi-scale sliding windows
    for cfg in WINDOW_CONFIGS:
        win = int(cfg["seconds"] * sr)
        stride = int(cfg["stride"] * sr)
        for start_idx in range(0, total_length - win, stride):
            end_idx = start_idx + win
            _score_chunk(
                audio_48k[start_idx:end_idx],
                start_idx / sr,
                end_idx / sr,
            )

    # Full-clip (CLAP's native 10 s input)
    _score_chunk(audio_48k, 0.0, total_length / sr)

    # Collect detected events
    detected = []
    for key in EVENT_KEYS:
        sc = best_score_per_event[key]
        if sc > EVENT_THRESHOLD:
            detected.append(
                {
                    "type": "sound_event",
                    "content": key.replace("_", " "),
                    "time": best_window_per_event[key],
                    "confidence": round(sc, 3),
                }
            )

    # Cap at top 2 highest confidence events to suppress FP inflation
    detected.sort(key=lambda x: x["confidence"], reverse=True)
    sound_events = detected[:2]

    return {
        "detected_language": detected_lang,
        "full_transcript": transcript,
        "speech": speech_segments,
        "events": sound_events,
    }
