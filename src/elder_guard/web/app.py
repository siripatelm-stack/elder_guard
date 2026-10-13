import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..inference import Detector
from ..speech import Transcriber

ROOT = Path(__file__).resolve().parents[3]                      # project root
EMBEDDER_DIR = Path(os.environ.get("EMBEDDER_DIR", ROOT / "models" / "distilbert"))
WEIGHTS = Path(os.environ.get("WEIGHTS_PATH", ROOT / "models" / "causal_lstm.pt"))
INDEX = Path(__file__).parent / "static" / "index.html"

for p in (EMBEDDER_DIR, WEIGHTS):
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Copy the files exported from Colab into models/ (see README).")

detector = Detector(EMBEDDER_DIR, WEIGHTS)
transcriber = Transcriber(os.environ.get("WHISPER_SIZE", "base"))
app = FastAPI(title="ElderGuard demo")


class Call(BaseModel):
    text: str


@app.get("/")
def home():
    return FileResponse(INDEX)


@app.post("/analyze")
def analyze(call: Call):
    return detector.score(call.text)


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    try:
        text, language = transcriber.transcribe_bytes(file.file.read(), suffix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    return {"text": text, "language": language}