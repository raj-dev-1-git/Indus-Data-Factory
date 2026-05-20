import subprocess
from pathlib import Path
import shutil

DATASET = Path("dataset/speech")
CLEAN = Path("clean")

LANGUAGES = ["marathi", "tamil", "bangla", "telugu"]


for lang in LANGUAGES:

    src_dir = DATASET / lang
    dst_dir = CLEAN / lang

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCleaning {lang}...")

    # copy transcript if exists
    tsv = src_dir / "line_index.tsv"
    if tsv.exists():
        shutil.copy(tsv, dst_dir / "line_index.tsv")

    for wav in src_dir.glob("*.wav"):

        out = dst_dir / wav.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "loudnorm",
            str(out),
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Done: {lang}")

print("\nAll speech cleaned.")
