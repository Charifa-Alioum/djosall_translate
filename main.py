from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil, os, uuid

from models.translation import translate_text
from models.stt import transcribe_audio
from models.tts import generate_tts_audio

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class TranslationInput(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class TTSInput(BaseModel):
    text: str

@app.post("/translate")
def translate(data: TranslationInput):
    try:
        result = translate_text(data.text, data.source_lang, data.target_lang)
        return {"translated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de traduction: {e}")

@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Fichier audio requis.")

    ext = os.path.splitext(file.filename)[-1]
    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = transcribe_audio(path)
        return {"transcription": text}
    finally:
        if os.path.exists(path): os.remove(path)

@app.post("/tts")
def tts(payload: TTSInput):
    try:
        audio_path = generate_tts_audio(payload.text)
        return FileResponse(audio_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur TTS: {e}")
