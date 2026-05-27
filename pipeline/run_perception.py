import json
from pathlib import Path

import librosa
import numpy as np
from faster_whisper import WhisperModel
from panns_inference import AudioTagging, labels as AUDIOSET_LABELS

BASE = Path(__file__).resolve().parent.parent

SCENES = BASE / "scenes" / "audio"
METADATA = BASE / "scenes" / "metadata"
OUT = BASE / "perception"

OUT.mkdir(parents=True, exist_ok=True)

# Clean old perception files
for old in OUT.glob("*.json"):
    old.unlink()

# =========================================================
# AUDIOSET CLASS MAPPING
# =========================================================

EVENT_TO_AUDIOSET = {
    "civil_defense_siren": ["Civil defense siren", "Siren"],
    "dog_bark": ["Bark", "Dog"],
    "gunfire": ["Gunshot, gunfire"],
    "subway_train": ["Subway, metro, underground"],
}

EVENT_KEYS = list(EVENT_TO_AUDIOSET.keys())

# =========================================================
# BUILD AUDIOSET INDICES
# =========================================================

_label_to_idx = {
    lab: i
    for i, lab in enumerate(AUDIOSET_LABELS)
}

EVENT_INDICES = {}

for key, audioset_names in EVENT_TO_AUDIOSET.items():

    indices = []

    for name in audioset_names:

        if name in _label_to_idx:

            indices.append(
                _label_to_idx[name]
            )

        else:

            print(
                f"WARNING: "
                f"AudioSet label '{name}' "
                f"not found for '{key}'"
            )

    EVENT_INDICES[key] = indices

print("\nEvent -> AudioSet index mapping:")

for key, idxs in EVENT_INDICES.items():

    mapped = [
        (AUDIOSET_LABELS[i], i)
        for i in idxs
    ]

    print(f"  {key:25s} -> {mapped}")

# =========================================================
# LANGUAGE MAP
# =========================================================

LANG_MAP = {
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
}

# =========================================================
# WINDOW CONFIGS
# =========================================================

WINDOW_CONFIGS = [
    {"seconds": 1.0, "stride": 0.5},
    {"seconds": 2.0, "stride": 1.0},
    {"seconds": 4.0, "stride": 2.0},
]

# =========================================================
# PANNs THRESHOLDS
# =========================================================

EVENT_THRESHOLDS = {
    "civil_defense_siren": 0.08,
    "dog_bark": 0.08,
    "gunfire": 0.05,
    "subway_train": 0.01,
}

# =========================================================
# WHISPER PROMPTS
# =========================================================

INITIAL_PROMPTS = {
    "mr": "",
    "ta": "",
    "te": ""
}

# =========================================================
# LOAD MODELS
# =========================================================

print("\nLoading Faster-Whisper...")

whisper_model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8"
)

print("\nLoading PANNs CNN14...")

panns_model = AudioTagging(
    checkpoint_path=None,
    device="cpu"
)

# =========================================================
# PROCESS FILES
# =========================================================

scene_files = sorted(
    SCENES.glob("*.wav")
)

for audio_path in scene_files:

    print(f"\nProcessing: {audio_path.name}")

    meta_file = (
        METADATA /
        f"{audio_path.stem}.json"
    )

    lang_code = None

    if meta_file.exists():

        with open(
            meta_file,
            "r",
            encoding="utf-8"
        ) as f:

            meta = json.load(f)

        lang_code = LANG_MAP.get(
            meta.get("language", ""),
            None
        )

    # =====================================================
    # ASR
    # =====================================================

    try:

        whisper_audio, _ = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        if lang_code:

            prompt = INITIAL_PROMPTS.get(
                lang_code,
                ""
            )

            segments, info = whisper_model.transcribe(
                whisper_audio,
                language=lang_code,
                beam_size=5,
                temperature=[
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0
                ],
                vad_filter=True,
                initial_prompt=prompt,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                repetition_penalty=1.2
            )

        else:

            segments, info = whisper_model.transcribe(
                whisper_audio,
                beam_size=5,
                temperature=[
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0
                ],
                vad_filter=True,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                repetition_penalty=1.2
            )

        segments = list(segments)

        transcript = " ".join(
            [seg.text for seg in segments]
        )

        safe_transcript = transcript.encode('ascii', 'backslashreplace').decode('ascii')
        print(
            f"Transcript: "
            f"{safe_transcript[:80]}"
        )

    except Exception as e:

        print(f"ASR failed: {e}")

        continue

    # =====================================================
    # SPEECH SEGMENTS
    # =====================================================

    speech_segments = []

    for seg in segments:

        speech_segments.append(
            {
                "type": "speech",
                "content": seg.text.strip(),
                "time": (
                    f"{seg.start:.2f}"
                    f"-{seg.end:.2f}"
                ),
                "confidence": round(
                    float(seg.avg_logprob),
                    3
                ),
            }
        )

    # =====================================================
    # EVENT DETECTION
    # =====================================================

    audio, sr = librosa.load(
        audio_path,
        sr=32000,
        mono=True
    )

    # =====================================================
    # HPSS SPEECH SUPPRESSION
    # =====================================================

    # harmonic, percussive = librosa.effects.hpss(
    #     audio
    # )

    # audio = percussive

    total_length = len(audio)

    best_score_per_event = {
        k: -999.0
        for k in EVENT_KEYS
    }

    best_window_per_event = {
        k: ""
        for k in EVENT_KEYS
    }

    # =====================================================
    # SCORING FUNCTION
    # =====================================================

    def score_chunk(
        chunk,
        t_start,
        t_end
    ):

        if np.max(np.abs(chunk)) < 0.01:
            return

        waveform = chunk[np.newaxis, :]

        clipwise_output, _ = (
            panns_model.inference(waveform)
        )

        probs = clipwise_output[0]

        for key in EVENT_KEYS:

            max_prob = max(
                float(probs[idx])
                for idx in EVENT_INDICES[key]
            )

            if max_prob > best_score_per_event[key]:

                best_score_per_event[key] = max_prob

                best_window_per_event[key] = (
                    f"{t_start:.2f}"
                    f"-{t_end:.2f}"
                )

    # =====================================================
    # MULTI-SCALE WINDOW SCANNING
    # =====================================================

    for cfg in WINDOW_CONFIGS:

        window_size = int(
            cfg["seconds"] * sr
        )

        stride_size = int(
            cfg["stride"] * sr
        )

        for start_idx in range(
            0,
            total_length - window_size,
            stride_size
        ):

            end_idx = (
                start_idx +
                window_size
            )

            chunk = audio[
                start_idx:end_idx
            ]

            score_chunk(
                chunk,
                start_idx / sr,
                end_idx / sr
            )

    # =====================================================
    # FULL CLIP PASS
    # =====================================================

    score_chunk(
        audio,
        0.0,
        total_length / sr
    )

    # =====================================================
    # DEBUG PRINTS
    # =====================================================

    print("\nBest event scores:")

    for key in EVENT_KEYS:

        print(
            f"{key:25s} "
            f"{best_score_per_event[key]:.3f} "
            f"@ "
            f"{best_window_per_event[key]}"
        )

    # =====================================================
    # EVENT DETECTION
    # =====================================================

    detected = []

    for key in EVENT_KEYS:

        score = best_score_per_event[key]

        threshold = EVENT_THRESHOLDS[key]

        if score >= threshold:

            detected.append(
                {
                    "type": "sound_event",
                    "content": (
                        key.replace("_", " ")
                    ),
                    "time": (
                        best_window_per_event[key]
                    ),
                    "confidence": round(
                        score,
                        3
                    ),
                }
            )

    # =====================================================
    # DYNAMIC FILTERING
    # =====================================================

    detected.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    if len(detected) > 0:

        top_conf = detected[0]["confidence"]

        sound_events = [
            d for d in detected
            if d["confidence"] >= (
                top_conf - 0.20
            )
        ]

    else:

        sound_events = []

    # =====================================================
    # FINAL JSON
    # =====================================================

    perception = {
        "audio_file": audio_path.name,
        "detected_language": (
            lang_code or info.language
        ),
        "full_transcript": transcript,
        "speech": speech_segments,
        "events": sound_events,
    }

    out_file = (
        OUT /
        f"{audio_path.stem}.json"
    )

    with open(
        out_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            perception,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved: {out_file.name}")

print("\nPerception complete.")