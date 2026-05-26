import os
import json
import random
from pathlib import Path

random.seed(42)


BASE = Path(__file__).resolve().parent.parent

CLEAN = BASE / "clean"
EVENTS = BASE / "dataset" / "events"

OUT_META = BASE / "scenes" / "metadata"

OUT_META.mkdir(parents=True, exist_ok=True)


SCENE_DURATION = 10.0

NUM_SCENES = 5

LANGUAGES = ["marathi", "tamil", "bangla", "telugu"]

# Added crowd
EVENT_TYPES = ["airplane", "dog_bark", "siren", "traffic", "crowd"]


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

    speech_files = list(speech_dir.glob("*.wav"))

    if not speech_files:
        continue

    transcripts = load_transcripts(speech_dir / "line_index.tsv")

    speech_file = random.choice(speech_files)

    # SPEECH TRACK

    tracks = [
        {
            "role": "speech",
            "source": str(speech_file.relative_to(BASE)),
            "start": round(random.uniform(2.0, 4.0), 2),
            "gain_db": 0.0,
            "transcript": transcripts.get(speech_file.name, ""),
        }
    ]

    # MULTI-EVENT SCENES

    # Lower complexity to realistic levels
    num_events = random.randint(1, 3)

    for _ in range(num_events):

        event_type = random.choice(EVENT_TYPES)

        event_dir = EVENTS / event_type

        event_files = list(event_dir.glob("*.wav"))

        if not event_files:
            continue

        event_file = random.choice(event_files)

        tracks.append(
            {
                "role": "event",
                "event_type": event_type,
                "source": str(event_file.relative_to(BASE)),
                "start": round(random.uniform(0.0, 8.0), 2),
                "gain_db": round(random.uniform(-35, -25), 2),
            }
        )

        # FINAL SCENE

    scene = {
        "scene_id": f"scene_{i:04d}",
        "language": lang,
        "duration": SCENE_DURATION,
        "tracks": tracks,
    }

    out_file = OUT_META / f"{scene['scene_id']}.json"

    with open(out_file, "w", encoding="utf-8") as f:

        json.dump(scene, f, indent=2, ensure_ascii=False)

    print(f"Generated: " f"{scene['scene_id']}")

    generated += 1

print(f"\nGenerated " f"{generated} valid scenes.")
