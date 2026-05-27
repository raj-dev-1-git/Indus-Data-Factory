"""Diagnostic: print PANNs CNN14 probabilities for every window."""
import json
from pathlib import Path

import librosa
import numpy as np
from panns_inference import AudioTagging, labels as AUDIOSET_LABELS

BASE = Path(__file__).resolve().parent.parent
SCENES = BASE / "scenes" / "audio"
METADATA = BASE / "scenes" / "metadata"

EVENT_TO_AUDIOSET = {
    "civil_defense_siren": ["Civil defense siren", "Siren"],
    "dog_bark": ["Bark", "Dog"],
    "gunfire": ["Gunshot, gunfire"],
    "subway_train": ["Subway, metro, underground"],
}

EVENT_KEYS = list(EVENT_TO_AUDIOSET.keys())

_label_to_idx = {lab: i for i, lab in enumerate(AUDIOSET_LABELS)}
EVENT_INDICES = {}
for key, names in EVENT_TO_AUDIOSET.items():
    EVENT_INDICES[key] = [_label_to_idx[n] for n in names if n in _label_to_idx]

WINDOW_SECONDS = 5
STRIDE_SECONDS = 1.5

print("Loading PANNs CNN14...")
panns_model = AudioTagging(checkpoint_path=None, device="cpu")

for audio_path in sorted(SCENES.glob("*.wav")):
    meta_file = METADATA / f"{audio_path.stem}.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    gt_events = [t["event_type"] for t in meta["tracks"] if t["role"] == "event"]

    print(f"\n{'='*70}")
    print(f"{audio_path.name}  |  GT events: {gt_events}")
    print(f"{'='*70}")

    audio, sr = librosa.load(audio_path, sr=32000, mono=True)
    window_size = int(WINDOW_SECONDS * sr)
    stride_size = int(STRIDE_SECONDS * sr)

    best_per_event = {k: (-999, "") for k in EVENT_KEYS}

    for start_idx in range(0, len(audio) - window_size, stride_size):
        end_idx = start_idx + window_size
        chunk = audio[start_idx:end_idx]

        if np.max(np.abs(chunk)) < 0.01:
            continue

        waveform = chunk[np.newaxis, :]
        (clipwise_output, _) = panns_model.inference(waveform)
        probs = clipwise_output[0]

        t_start = start_idx / sr
        t_end = end_idx / sr

        scores = {}
        for key in EVENT_KEYS:
            scores[key] = max(float(probs[idx]) for idx in EVENT_INDICES[key])

        scores_str = "  ".join(
            f"{k[:8]:>8}={s:.3f}" for k, s in scores.items()
        )
        print(f"  [{t_start:5.1f}-{t_end:5.1f}s]  {scores_str}")

        for k, s in scores.items():
            if s > best_per_event[k][0]:
                best_per_event[k] = (s, f"{t_start:.1f}-{t_end:.1f}")

    print(f"\n  BEST scores per event:")
    for k in EVENT_KEYS:
        score, window = best_per_event[k]
        marker = " <<< GT" if k in gt_events else ""
        print(f"    {k:25s}  {score:.4f}  @ {window}{marker}")
