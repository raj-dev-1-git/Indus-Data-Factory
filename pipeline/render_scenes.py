import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent

META_DIR = BASE / "scenes" / "metadata"
OUT_AUDIO = BASE / "scenes" / "audio"

OUT_AUDIO.mkdir(parents=True, exist_ok=True)


SR = 16000
SCENE_DURATION = 10
SCENE_SAMPLES = SR * SCENE_DURATION


def load_audio(path):

    y, _ = librosa.load(path, sr=SR, mono=True)

    return y


scene_files = sorted(META_DIR.glob("*.json"))

for meta_file in tqdm(scene_files):

    with open(meta_file, "r", encoding="utf-8") as f:
        scene = json.load(f)

    mix = np.zeros(SCENE_SAMPLES, dtype=np.float32)

    for track in scene["tracks"]:

        audio_path = BASE / track["source"]

        y = load_audio(audio_path)

        gain = 10 ** (track["gain_db"] / 20)

        y = y * gain

        start_sample = int(track["start"] * SR)

        end_sample = min(start_sample + len(y), SCENE_SAMPLES)

        clip_len = end_sample - start_sample

        mix[start_sample:end_sample] += y[:clip_len]

    # normalize
    peak = np.max(np.abs(mix))

    if peak > 0:
        mix = mix / peak

    out_path = OUT_AUDIO / f"{scene['scene_id']}.wav"

    sf.write(out_path, mix, SR)

    print(f"Rendered: {scene['scene_id']}")

print("\nAll scenes rendered.")
