import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request

from fastapi.responses import JSONResponse, HTMLResponse

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

from sqlalchemy.orm import Session

from app.database import SessionLocal

from app.models import AudioResult

from app.services.perception_service import run_perception

from app.database import engine
from app.models import AudioResult

AudioResult.metadata.create_all(bind=engine)

app = FastAPI()



APP_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


BASE = Path(__file__).resolve().parent.parent

UPLOADS = BASE / "uploads"

UPLOADS.mkdir(parents=True, exist_ok=True)


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
                except:
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
def upload_audio(file: UploadFile = File(...)):

    # VALIDATE

    if not file.filename.endswith(".wav"):

        return JSONResponse(
            status_code=400, content={"error": "Only .wav files allowed"}
        )

        # SAVE AUDIO

    safe_name = Path(file.filename).name
    save_path = UPLOADS / safe_name

    with open(save_path, "wb") as f:

        content = file.file.read()

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
        "filename": file.filename,
        "perception": perception,
        "reasoning": reasoning,
        "database_id": audio_result.id,
    }
