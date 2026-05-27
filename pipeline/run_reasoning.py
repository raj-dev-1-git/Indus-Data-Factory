import json
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

PERCEPTION = BASE / "perception"

REASONING = BASE / "reasoning"

PROMPT_FILE = BASE / "archive" / "prompts" / "prompt_v1.txt"

REASONING.mkdir(parents=True, exist_ok=True)

# Clean old reasoning files
for old in REASONING.glob("*.json"):
    old.unlink()


OLLAMA_EXE = os.environ.get("OLLAMA_EXE", "ollama")

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")


def _load_system_prompt():
    """Load the system prompt from disk, with a sensible fallback."""
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8")
    print(f"WARNING: Prompt file not found at {PROMPT_FILE}, using default prompt.")
    return (
        "You are an Audio Language Model reasoning engine. "
        "Analyze the provided perception JSON and output STRICT JSON with keys: "
        "summary, environment, events, inference, risk_level."
    )


SYSTEM_PROMPT = _load_system_prompt()


json_files = sorted(PERCEPTION.glob("*.json"))

for perception_file in json_files:

    print(f"\nReasoning: " f"{perception_file.name}")

    with open(perception_file, "r", encoding="utf-8") as f:

        perception_data = json.load(f)

        # BUILD PROMPT

    full_prompt = f"""
{SYSTEM_PROMPT}

Perception JSON:

{json.dumps(
    perception_data,
    indent=2,
    ensure_ascii=False
)}
"""

    # RUN MODEL

    result = subprocess.run(
        [OLLAMA_EXE, "run", MODEL],
        input=full_prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    output = result.stdout.strip()

    output = output.replace("```json", "")

    output = output.replace("```", "").strip()

    # EMPTY OUTPUT

    if output == "":

        print("\nMODEL ERROR:\n")

        print(result.stderr)

        output = json.dumps(
            {
                "summary": "No reasoning generated.",
                "environment": "Unknown",
                "inference": "Reasoning model failed.",
                "risk_level": "Unknown",
            }
        )

        # VALIDATE JSON

    try:

        parsed = json.loads(output)

    except Exception:

        parsed = {
            "summary": "Reasoning parsing failed.",
            "environment": "Unknown",
            "inference": output,
            "risk_level": "Unknown",
        }

        # SAVE

    out_file = REASONING / perception_file.name

    with open(out_file, "w", encoding="utf-8") as f:

        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"Saved: " f"{out_file.name}")

print("\nReasoning complete.")
