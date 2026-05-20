# main.py

import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent


def run_step(script_name):

    print(f"Running: {script_name}")

    result = subprocess.run(
        [".venv\\Scripts\\python.exe", str(BASE / "pipeline" / script_name)], text=True
    )

    # CHECK FAILURE

    if result.returncode != 0:

        print(f"\nFAILED: " f"{script_name}")

        exit(1)

    print(f"\nCOMPLETED: " f"{script_name}")


PIPELINE = [
    "generate_mixer_logs.py",
    "render_scenes.py",
    "run_perception.py",
    "run_reasoning.py",
    "evaluate_asr.py",
]


print("\nSTARTING AUDIO " "INTELLIGENCE PIPELINE\n")

for step in PIPELINE:

    run_step(step)

print("\nPIPELINE COMPLETE.\n")
