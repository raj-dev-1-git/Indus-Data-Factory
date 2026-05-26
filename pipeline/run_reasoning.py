import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

PERCEPTION = BASE / "perception"

REASONING = BASE / "reasoning"

import os

OLLAMA_EXE = os.environ.get("OLLAMA_EXE", "ollama")

MODEL = "qwen2.5:3b"

REASONING.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """You are an advanced audio intelligence reasoning system.
Given a JSON payload describing detected speech and sound events from an audio clip, you must provide:
1. A concise summary of what is happening.
2. The most likely environment or setting.
3. Logical inferences about the situation.
4. A risk level assessment (Low, Medium, High).

Respond ONLY with a valid JSON object matching this schema:
{
  "summary": "string",
  "environment": "string",
  "inference": "string",
  "risk_level": "string"
}
"""


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
