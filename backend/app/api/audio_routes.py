from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.whisper_stt import transcribe_audio 
import os
import uuid

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/transcribe_audio/")
async def transcribe_audio_route(file: UploadFile = File(...)):
    file_path = None
    try:
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        text = transcribe_audio(file_path)
        os.remove(file_path)
        return {"status": "success", "transcription": text}
    except Exception as e:
        if file_path and os.path.exists(file_path): 
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
