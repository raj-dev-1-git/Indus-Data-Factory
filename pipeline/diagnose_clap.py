"""Diagnostic: print CLAP cosine-similarity scores for every window."""
import json
from pathlib import Path

import librosa
import numpy as np
import laion_clap

BASE = Path(__file__).resolve().parent.parent
SCENES = BASE / "scenes" / "audio"
METADATA = BASE / "scenes" / "metadata"

EVENT_PROMPTS = {
    "car_honk":              "a car horn honking on the road",
    "civil_defense_siren":   "a civil defense siren or air raid siren wailing",
    "dog_bark":              "a dog barking loudly",
    "explosion":             "a loud explosion or blast",
    "fighter_jet_engine":    "a fighter jet engine roaring overhead",
    "gunfire":               "the sound of gunfire or gunshots",
    "subway_train":          "a subway train or metro train passing",
}

EVENT_KEYS  = list(EVENT_PROMPTS.keys())
EVENT_TEXTS = list(EVENT_PROMPTS.values())

WINDOW_SECONDS = 5
STRIDE_SECONDS = 1.5

print("Loading CLAP (HTSAT-base, no fusion)...")
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
clap_model.load_ckpt(ckpt=str(BASE / "music_speech_audioset_epoch_15_esc_89.98.pt"))

# Text embeddings
text_embed = clap_model.get_text_embedding(EVENT_TEXTS, use_tensor=False)
text_embed = text_embed / np.linalg.norm(text_embed, axis=-1, keepdims=True)

for audio_path in sorted(SCENES.glob("*.wav")):
    meta_file = METADATA / f"{audio_path.stem}.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    gt_events = [t["event_type"] for t in meta["tracks"] if t["role"] == "event"]

    print(f"\n{'='*70}")
    print(f"{audio_path.name}  |  GT events: {gt_events}")
    print(f"{'='*70}")

    audio, sr = librosa.load(audio_path, sr=48000, mono=True)
    window_size = int(WINDOW_SECONDS * sr)
    stride_size = int(STRIDE_SECONDS * sr)

    best_per_event = {k: (-999, "") for k in EVENT_KEYS}

    for start_idx in range(0, len(audio) - window_size, stride_size):
        end_idx = start_idx + window_size
        chunk = audio[start_idx:end_idx]

        if np.max(np.abs(chunk)) < 0.01:
            continue

        audio_embed = clap_model.get_audio_embedding_from_data(
            x=[chunk], use_tensor=False
        )
        audio_embed = audio_embed / np.linalg.norm(audio_embed, axis=-1, keepdims=True)
        similarity = (audio_embed @ text_embed.T)[0]

        t_start = start_idx / sr
        t_end = end_idx / sr
        scores_str = "  ".join(
            f"{k[:8]:>8}={s:+.3f}" for k, s in zip(EVENT_KEYS, similarity)
        )
        print(f"  [{t_start:5.1f}-{t_end:5.1f}s]  {scores_str}")

        for k, s in zip(EVENT_KEYS, similarity):
            if s > best_per_event[k][0]:
                best_per_event[k] = (s, f"{t_start:.1f}-{t_end:.1f}")

    print(f"\n  BEST scores per event:")
    for k in EVENT_KEYS:
        score, window = best_per_event[k]
        marker = " <<< GT" if k in gt_events else ""
        print(f"    {k:25s}  {score:+.4f}  @ {window}{marker}")
