import json
from pathlib import Path

import librosa
import numpy as np
from faster_whisper import WhisperModel
import laion_clap

BASE = Path(__file__).resolve().parent.parent

SCENES = BASE / "scenes" / "audio"
METADATA = BASE / "scenes" / "metadata"
OUT = BASE / "perception"

OUT.mkdir(parents=True, exist_ok=True)

# Clean old perception files
for old in OUT.glob("*.json"):
    old.unlink()

# =========================================================
# EVENT PROMPTS
# =========================================================

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

# =========================================================
# FLATTEN PROMPTS
# =========================================================

_ALL_PROMPTS = []

_PROMPT_TO_EVENT_IDX = []

for i, key in enumerate(EVENT_KEYS):

    for prompt in EVENT_PROMPTS[key]:

        _ALL_PROMPTS.append(prompt)

        _PROMPT_TO_EVENT_IDX.append(i)

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
    {"seconds": 1.0, "stride": 0.25},
    {"seconds": 2.0, "stride": 0.5},
    {"seconds": 4.0, "stride": 1.0},
]

# =========================================================
# EVENT THRESHOLDS
# =========================================================

EVENT_THRESHOLDS = {
    "car_honk": 0.18,
    "civil_defense_siren": 0.16,
    "dog_bark": 0.17,
    "explosion": 0.22,
    "fighter_jet_engine": 0.15,
    "gunfire": 0.22,
    "subway_train": 0.14,
}

# =========================================================
# WHISPER PROMPTS
# =========================================================

INITIAL_PROMPTS = {
    "mr": "कृपया सर्व आकडे शब्दांत लिहा.",
    "ta": "தயவுசெய்து அனைத்து எண்களையும் வார்த்தைகளில் எழுதவும்.",
    "te": "దయచేసి అన్ని సంఖ్యలను పదాలలో రాయండి."
}

# =========================================================
# LOAD MODELS
# =========================================================

print("Loading Faster-Whisper...")

whisper_model = WhisperModel(
    "large-v3-turbo",
    device="cpu",
    compute_type="int8"
)

print("Loading CLAP...")

clap_model = laion_clap.CLAP_Module(
    enable_fusion=False,
    amodel='HTSAT-base'
)

clap_model.load_ckpt(
    ckpt=str(
        BASE / "music_speech_audioset_epoch_15_esc_89.98.pt"
    )
)

# =========================================================
# PRECOMPUTE TEXT EMBEDDINGS
# =========================================================

text_embed = clap_model.get_text_embedding(
    _ALL_PROMPTS,
    use_tensor=False
)

text_embed = text_embed / np.linalg.norm(
    text_embed,
    axis=-1,
    keepdims=True
)

# =========================================================
# PROCESS FILES
# =========================================================

scene_files = sorted(SCENES.glob("*.wav"))

for audio_path in scene_files:

    print(f"\nProcessing: {audio_path.name}")

    meta_file = METADATA / f"{audio_path.stem}.json"

    lang_code = None

    if meta_file.exists():

        with open(meta_file, "r", encoding="utf-8") as f:

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

        audio_for_whisper = whisper_audio

        if lang_code:

            prompt = INITIAL_PROMPTS.get(
                lang_code,
                ""
            )

            segments, info = whisper_model.transcribe(
                audio_for_whisper,
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
                audio_for_whisper,
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

        print(
            f"Transcript: "
            f"{transcript[:80]}"
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
                "time": f"{seg.start:.2f}-{seg.end:.2f}",
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
        sr=48000,
        mono=True
    )

    # =====================================================
    # SIMPLE SPEECH SUPPRESSION
    # =====================================================

    harmonic, percussive = librosa.effects.hpss(audio)

    # keep environmental/percussive part
    audio = percussive

    total_length = len(audio)

    best_score_per_event = {
        k: -999.0
        for k in EVENT_KEYS
    }

    best_window_per_event = {
        k: ""
        for k in EVENT_KEYS
    }

    def score_chunk(chunk, t_start, t_end):

        if np.max(np.abs(chunk)) < 0.01:
            return

        audio_embed = clap_model.get_audio_embedding_from_data(
            x=[chunk],
            use_tensor=False
        )

        audio_embed = audio_embed / np.linalg.norm(
            audio_embed,
            axis=-1,
            keepdims=True
        )

        similarity = (audio_embed @ text_embed.T)[0]

        for ei, key in enumerate(EVENT_KEYS):

            prompt_scores = [
                float(similarity[pi])
                for pi, eidx in enumerate(_PROMPT_TO_EVENT_IDX)
                if eidx == ei
            ]

            max_score = max(prompt_scores)

            if max_score > best_score_per_event[key]:

                best_score_per_event[key] = max_score

                best_window_per_event[key] = (
                    f"{t_start:.2f}-{t_end:.2f}"
                )

    # =====================================================
    # MULTI-SCALE WINDOWS
    # =====================================================

    for cfg in WINDOW_CONFIGS:

        window_size = int(cfg["seconds"] * sr)

        stride_size = int(cfg["stride"] * sr)

        for start_idx in range(
            0,
            total_length - window_size,
            stride_size
        ):

            end_idx = start_idx + window_size

            chunk = audio[start_idx:end_idx]

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
            f"@ {best_window_per_event[key]}"
        )

    # =====================================================
    # DETECTION
    # =====================================================

    detected = []

    for key in EVENT_KEYS:

        score = best_score_per_event[key]

        threshold = EVENT_THRESHOLDS[key]

        if score >= threshold:

            detected.append(
                {
                    "type": "sound_event",
                    "content": key.replace("_", " "),
                    "time": best_window_per_event[key],
                    "confidence": round(score, 3),
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
            if d["confidence"] >= top_conf - 0.08
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

    out_file = OUT / f"{audio_path.stem}.json"

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