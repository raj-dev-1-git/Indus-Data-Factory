import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent

import os

OLLAMA_EXE = os.environ.get("OLLAMA_EXE", "ollama")
MODEL = "qwen2.5:3b"

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
