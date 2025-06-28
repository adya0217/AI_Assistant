from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services.classroom_image_analyzer import ClassroomImageAnalyzer  
from app.services.context_manager import context_manager
import os
from typing import Dict, List
import uuid
import logging

router = APIRouter()
image_analyzer = ClassroomImageAnalyzer()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...), text: str = Form(None), questions: List[str] = Form(None)) -> Dict:
    file_path = None
    try:
        logger.info(f"/analyze endpoint called. Filename: {file.filename}, Text: {text}, Questions: {questions}")
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        image_id = str(uuid.uuid4())

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        analysis_result = image_analyzer.analyze_image_comprehensive(file_path, questions, user_text=text)
        os.remove(file_path)

        
        context_manager.add_image_analysis(image_id, analysis_result)

        t
        llm_answer = None
        if text:
            from app.services.llm_chain import get_llm_response
            llm_answer = get_llm_response(text, analysis_result.get('explanation', ''))

        return {
            "status": "success",
            "image_id": image_id,
            "analysis": analysis_result,
            "explanation": analysis_result.get('explanation', ''),
            "llm_answer": llm_answer
        }

    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {e}")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
