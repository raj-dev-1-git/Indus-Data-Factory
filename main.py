# main.py

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(script_name):

    print("\n" + "=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    start = time.time()

    result = subprocess.run(
        [
            PYTHON,
            str(BASE / "pipeline" / script_name),
        ],
        text=True,
    )

    # CHECK FAILURE
    if result.returncode != 0:

        print("\n" + "=" * 60)
        print(f"FAILED: {script_name}")
        print("=" * 60)

        sys.exit(1)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"COMPLETED: {script_name}")
    print(f"TIME: {elapsed:.2f} sec")
    print("=" * 60)


PIPELINE = [
    "generate_mixer_logs.py",
    "render_scenes.py",
    "run_perception.py",
    "evaluate_asr.py",
    "run_reasoning.py",
]


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("STARTING AUDIO INTELLIGENCE PIPELINE")
    print("=" * 60)

    pipeline_start = time.time()

    for step in PIPELINE:

        run_step(step)

    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"TOTAL TIME: {pipeline_elapsed:.2f} sec")
    print("=" * 60)