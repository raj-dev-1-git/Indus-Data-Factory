import os
import json
import random
import time
from pathlib import Path

# Use a unique seed each run (printed so you can reproduce if needed)
SEED = int(time.time() * 1000) % (2**32)
random.seed(SEED)
print(f"Random seed: {SEED}")

BASE = Path(__file__).resolve().parent.parent

CLEAN = BASE / "clean"
EVENTS = BASE / "dataset" / "events"

OUT_META = BASE / "scenes" / "metadata"
OUT_AUDIO = BASE / "scenes" / "audio"

OUT_META.mkdir(parents=True, exist_ok=True)
OUT_AUDIO.mkdir(parents=True, exist_ok=True)

# Clean old scenes so stale data doesn't carry over
for old in OUT_META.glob("*.json"):
    old.unlink()
for old in OUT_AUDIO.glob("*.wav"):
    old.unlink()
print("Cleaned old scene files.")

SCENE_DURATION = 10.0
NUM_SCENES = 20

LANGUAGES = [
    "marathi",
    "tamil",
    "telugu",
]

# aligned with dataset/events folder names
EVENT_TYPES = [
    "car_honk",
    "civil_defense_siren",
    "dog_bark",
    "explosion",
    "fighter_jet_engine",
    "gunfire",
    "subway_train",
]


def load_transcripts(tsv_path):

    mapping = {}

    if not tsv_path.exists():
        return mapping

    with open(tsv_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(maxsplit=1)

            if len(parts) != 2:
                continue

            audio_id, text = parts

            if not audio_id.endswith(".wav"):
                audio_id += ".wav"

            mapping[audio_id] = text

    return mapping


generated = 0

for i in range(NUM_SCENES):

    lang = random.choice(LANGUAGES)

    speech_dir = CLEAN / lang

    speech_files = list(
        speech_dir.glob("*.wav")
    )

    if not speech_files:
        continue

    transcripts = load_transcripts(
        speech_dir / "line_index.tsv"
    )

    # KEEP ONLY LONGER TRANSCRIPTS
    speech_files = [
        f for f in speech_files
        if len(
            transcripts.get(
                f.name,
                ""
            ).split()
        ) >= 5
    ]

    # no valid files after filtering
    if not speech_files:
        continue

    speech_file = random.choice(
        speech_files
    )

    # =========================
    # SPEECH TRACK
    # =========================

    tracks = [
        {
            "role": "speech",
            "source": str(
                speech_file.relative_to(BASE)
            ),
            "start": round(
                random.uniform(2.0, 4.0),
                2
            ),
            "gain_db": 0.0,
            "transcript": transcripts.get(
                speech_file.name,
                ""
            ),
        }
    ]

    # =========================
    # EVENTS
    # =========================

    # simpler scenes
    num_events = 1

    for _ in range(num_events):

        event_type = random.choice(
            EVENT_TYPES
        )

        event_dir = EVENTS / event_type

        event_files = list(
            event_dir.glob("*.wav")
        )

        if not event_files:
            continue

        event_file = random.choice(
            event_files
        )

        tracks.append(
            {
                "role": "event",
                "event_type": event_type,
                "source": str(
                    event_file.relative_to(BASE)
                ),
                "start": round(
                    random.uniform(0.0, 8.0),
                    2
                ),

                # SOFTER EVENTS to preserve speech SNR
                "gain_db": round(
                    random.uniform(-10, -4),
                    2
                ),
            }
        )

    # =========================
    # FINAL SCENE
    # =========================

    scene = {
        "scene_id": f"scene_{i:04d}",
        "language": lang,
        "duration": SCENE_DURATION,
        "tracks": tracks,
    }

    out_file = (
        OUT_META /
        f"{scene['scene_id']}.json"
    )

    with open(
        out_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            scene,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Generated: "
        f"{scene['scene_id']}"
    )

    generated += 1

print(
    f"\nGenerated "
    f"{generated} valid scenes."
)