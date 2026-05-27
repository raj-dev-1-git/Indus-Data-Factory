import json
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent

PROMPT_FILE = BASE / "archive" / "prompts" / "prompt_v1.txt"

OLLAMA_EXE = os.environ.get("OLLAMA_EXE", "ollama")
MODEL = "qwen2.5:1.5b"


with open(PROMPT_FILE, "r", encoding="utf-8") as f:

    SYSTEM_PROMPT = f.read()


def run_reasoning(perception_data):

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

    # RUN QWEN

    result = subprocess.run(
        [OLLAMA_EXE, "run", MODEL],
        input=full_prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    # OUTPUT

    output = result.stdout.strip()

    # CLEAN MARKDOWN

    output = output.replace("```json", "")

    output = output.replace("```", "").strip()

    # EMPTY RESPONSE

    if output == "":

        print("\nERROR FROM OLLAMA:\n")

        print(result.stderr)

        return {
            "summary": "No response from model.",
            "environment": "Unknown",
            "inference": "Reasoning model failed.",
            "risk_level": "Unknown",
        }

        # PARSE JSON

    try:

        parsed = json.loads(output)

        return parsed

        # FALLBACK

    except Exception:

        return {
            "summary": "Reasoning parsing failed.",
            "environment": "Unknown",
            "inference": output,
            "risk_level": "Unknown",
        }
