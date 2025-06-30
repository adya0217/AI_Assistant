from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services.classroom_image_analyzer import ClassroomImageAnalyzer  
from app.services.context_manager import context_manager
import os
from typing import Dict, List
import uuid
import logging
import time

router = APIRouter()
image_analyzer = ClassroomImageAnalyzer()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...), text: str = Form(None), questions: List[str] = Form(None)) -> Dict:
    start_time = time.time()
    file_path = None
    logger.info(f"[analyze_image] Request received: {file.filename}, Text: {text}, Questions: {questions}")
    try:
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        image_id = str(uuid.uuid4())
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info("[analyze_image] File saved, starting analysis...")
        analysis_start = time.time()
        analysis_result = image_analyzer.analyze_image_comprehensive(file_path, questions, user_text=text)
        analysis_time = time.time() - analysis_start
        logger.info(f"[analyze_image] Image analysis in {analysis_time:.2f}s")
        os.remove(file_path)
        logger.info(f"[analyze_image] Temp file {file_path} removed.")
        context_manager.add_image_analysis(image_id, analysis_result)
        llm_answer = None
        if text:
            from app.services.llm_chain import get_llm_response
            llm_start = time.time()
            logger.info("[analyze_image] Calling LLM for explanation...")
            llm_answer = get_llm_response(text, analysis_result.get('explanation', ''))
            llm_time = time.time() - llm_start
            logger.info(f"[analyze_image] LLM response in {llm_time:.2f}s")
        total_time = time.time() - start_time
        logger.info(f"[analyze_image] Sending response in {total_time:.2f}s")
        return {
            "status": "success",
            "image_id": image_id,
            "analysis": analysis_result,
            "explanation": analysis_result.get('explanation', ''),
            "llm_answer": llm_answer
        }
    except Exception as e:
        logger.error(f"[analyze_image] Error: {e}", exc_info=True)
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[analyze_image] Temp file {file_path} removed after error.")
        raise HTTPException(status_code=500, detail=str(e))
