from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import uuid
import os
from datetime import datetime

from app.services.whisper_stt import transcribe_audio
from app.services.classroom_image_analyzer import ClassroomImageAnalyzer
from app.services.llm_chain import multimodal_chain
from app.services.context_manager import context_manager

logger = logging.getLogger(__name__)
router = APIRouter()


image_analyzer = ClassroomImageAnalyzer()

class UnifiedRequest(BaseModel):
    
    text_query: Optional[str] = None
    voice_file: Optional[UploadFile] = None
    image_file: Optional[UploadFile] = None
    custom_prompt: Optional[str] = None
    include_analysis: bool = True

class UnifiedResponse(BaseModel):
    """Response model for unified multimodal analysis"""
    response: str
    input_types: List[str]
    analysis_summary: Dict[str, Any]
    session_id: str
    timestamp: str

class UnifiedCapabilitiesResponse(BaseModel):
    """Response model for capabilities endpoint"""
    capabilities: Dict[str, Any]
    context_management: Dict[str, bool]
    models_used: Dict[str, Any]

def sanitize_input(text: str) -> str:
    """Sanitize and normalize incoming text"""
    if not text:
        return ""
    
    
    cleaned = text.strip()
    
    cleaned = cleaned.replace('\n', ' ')
    
    cleaned = ' '.join(cleaned.split())
    
    cleaned = cleaned[:1000] if len(cleaned) > 1000 else cleaned
    
    return cleaned

@router.post("/analyze", response_model=UnifiedResponse)
async def unified_analyze(
    text_query: Optional[str] = Form(None),
    voice_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    custom_prompt: Optional[str] = Form(None),
    include_analysis: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    """
    Unified multimodal analysis endpoint that can handle:
    - Text input
    - Voice input (Whisper → Text)
    - Image input (OpenCV/YOLO/VIT → Analysis)
    - Combined multimodal input
    """
    
    session_id = str(uuid.uuid4())
    temp_files = []
    
    try:
        logger.info(f"Starting unified analysis session: {session_id}")
        
        
        input_types = []
        voice_transcription = None
        image_analysis = None
        final_query = sanitize_input(text_query) if text_query else ""
        
        
        if voice_file:
            logger.info("Processing voice input...")
            input_types.append("voice")
            
            
            temp_voice_path = f"temp_voice_{session_id}.wav"
            temp_files.append(temp_voice_path)
            
            contents = await voice_file.read()
            with open(temp_voice_path, "wb") as f:
                f.write(contents)
            
           
            voice_transcription = transcribe_audio(temp_voice_path)
            logger.info(f"Voice transcription: {voice_transcription}")
            
           
            clean_transcription = sanitize_input(voice_transcription)
            
          
            context_manager.add_voice_transcription(session_id, clean_transcription)
            
            
            if not final_query:
                final_query = clean_transcription
        
        
        if image_file:
            logger.info("Processing image input...")
            input_types.append("image")
            
            
            temp_image_path = f"temp_image_{session_id}.jpg"
            temp_files.append(temp_image_path)
            
            contents = await image_file.read()
            with open(temp_image_path, "wb") as f:
                f.write(contents)
            
            
            image_analysis = image_analyzer.analyze_image_comprehensive(temp_image_path)

            
            similar_images = []
            image_embedding = image_analyzer.get_image_embedding(image_analyzer.preprocess_image(temp_image_path))
            if image_embedding.size > 0:
                embedding_list = image_embedding.tolist()
                multimodal_chain.store_image_embedding(temp_image_path, embedding_list)
                similar_images = multimodal_chain.retrieve_similar_images(embedding_list)
                logger.info(f"Found {len(similar_images)} similar images.")
            else:
                logger.warning("Failed to generate image embedding.")
            
            if 'error' in image_analysis:
                raise HTTPException(status_code=500, detail=image_analysis['error'])
            
            
            image_id = str(uuid.uuid4())
            context_manager.add_image_analysis(image_id, image_analysis)
            
            logger.info(f"Image analysis completed: {len(image_analysis.get('objects', []))} objects detected")
        
        
        if text_query:
            input_types.append("text")
        
        
            chain_input_type = "multimodal"
        elif len(input_types) == 1:
            chain_input_type = input_types[0]
        else:
            raise HTTPException(status_code=400, detail="No input provided")
        
        
        if custom_prompt:
            final_query = sanitize_input(custom_prompt)
        
        
        if not final_query:
            raise HTTPException(status_code=400, detail="No valid query provided")
        
        
        logger.info(f"Processing with {chain_input_type} chain...")
        response = multimodal_chain.process_multimodal_input(
            query=final_query,
            voice_transcription=voice_transcription,
            image_analysis=image_analysis,
            input_type=chain_input_type
        )
        
        
        analysis_summary = {
            "input_types": input_types,
            "chain_type": chain_input_type,
            "voice_transcription": voice_transcription,
            "image_objects_detected": len(image_analysis.get('objects', [])) if image_analysis else 0,
            "image_subject": image_analysis.get('subject', 'none') if image_analysis else 'none',
            "similar_images": similar_images if 'similar_images' in locals() else [],
            "context_summary": context_manager.get_context_summary()
        }
        
        
        unified_response = UnifiedResponse(
            response=response,
            input_types=input_types,
            analysis_summary=analysis_summary,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Unified analysis completed for session {session_id}")
        return unified_response

    except Exception as e:
        logger.error(f"Error in unified analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")

@router.post("/stream")
async def unified_stream(
    text_query: Optional[str] = Form(None),
    voice_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    """
    Streaming version of unified analysis for real-time responses
    """
    
    return await unified_analyze(text_query, voice_file, image_file)

@router.get("/capabilities", response_model=UnifiedCapabilitiesResponse)
async def get_capabilities():
    """
    Get information about available multimodal capabilities
    """
    return UnifiedCapabilitiesResponse(
        capabilities={
            "text_processing": {
                "description": "Process text queries with context awareness",
                "endpoint": "/analyze",
                "input": "text_query parameter"
            },
            "voice_processing": {
                "description": "Process voice input using Whisper STT",
                "endpoint": "/analyze", 
                "input": "voice_file parameter",
                "models": ["Whisper"]
            },
            "image_processing": {
                "description": "Process images using OpenCV, YOLO, and ViT",
                "endpoint": "/analyze",
                "input": "image_file parameter", 
                "models": ["OpenCV", "YOLO", "ViT", "BLIP", "TrOCR"]
            },
            "multimodal_processing": {
                "description": "Combine multiple input types for comprehensive analysis",
                "endpoint": "/analyze",
                "input": "Multiple parameters",
                "context_awareness": True
            }
        },
        context_management={
            "conversation_history": True,
            "image_context": True,
            "voice_context": True,
            "session_management": True
        },
        models_used={
            "llm": "Ollama Mistral",
            "stt": "Whisper",
            "vision": ["YOLO", "ViT", "BLIP", "TrOCR"],
            "embeddings": "all-MiniLM-L6-v2"
        }
    )
