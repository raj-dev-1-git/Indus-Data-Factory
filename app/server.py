import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request

from fastapi.responses import JSONResponse, HTMLResponse

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

from sqlalchemy.orm import Session

from app.database import SessionLocal

from app.models import AudioResult

from app.services.perception_service import run_perception

from app.services.reasoning_service import run_reasoning

# Resolve paths relative to this file so the server works from any cwd
_APP_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(_APP_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(_APP_DIR / "templates"))


BASE = _APP_DIR.parent

UPLOADS = BASE / "uploads"

UPLOADS.mkdir(parents=True, exist_ok=True)


def _safe_filename(filename: str) -> str:
    """Strip path separators and dangerous characters from an uploaded filename."""
    # Take only the basename (ignore directory components)
    name = Path(filename).name
    # Remove anything that is not alphanumeric, hyphen, underscore, or dot
    name = re.sub(r"[^\w\-.]", "_", name)
    return name or "upload.wav"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.get("/health")
def health():

    return {"status": "ok"}


@app.get("/history")
def get_history():
    db: Session = SessionLocal()
    try:
        records = db.query(AudioResult).order_by(AudioResult.id.desc()).limit(50).all()
        history_list = []
        for r in records:
            reasoning_data = {}
            if r.reasoning:
                try:
                    reasoning_data = json.loads(r.reasoning)
                except Exception:
                    reasoning_data = {"summary": "Error parsing reasoning"}
            
            history_list.append({
                "id": r.id,
                "filename": r.filename,
                "transcript": r.transcript,
                "summary": reasoning_data.get("summary", ""),
                "risk_level": reasoning_data.get("risk_level", "Unknown")
            })
        return {"history": history_list}
    finally:
        db.close()


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):

    # VALIDATE

    if not file.filename.endswith(".wav"):

        return JSONResponse(
            status_code=400, content={"error": "Only .wav files allowed"}
        )

    # SAVE AUDIO

    safe_name = _safe_filename(file.filename)
    save_path = UPLOADS / safe_name

    with open(save_path, "wb") as f:

        content = await file.read()

        f.write(content)

    # RUN PERCEPTION

    perception = run_perception(save_path)

    # RUN REASONING

    reasoning = run_reasoning(perception)

    # SQLITE
    db: Session = SessionLocal()

    try:
        audio_result = AudioResult(
            filename=safe_name,
            transcript=perception["full_transcript"],
            reasoning=json.dumps(reasoning, ensure_ascii=False)
        )

        db.add(audio_result)

        db.commit()

        db.refresh(audio_result)

    finally:
        db.close()

    # RESPONSE

    return {
        "message": "Inference complete",
        "filename": safe_name,
        "perception": perception,
        "reasoning": reasoning,
        "database_id": audio_result.id,
    }
