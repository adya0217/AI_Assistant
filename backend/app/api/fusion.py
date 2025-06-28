import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import logging
from datetime import datetime, timedelta
import uuid
from typing import Optional, List
from app.services.whisper_stt import record_audio, transcribe_audio
from app.services.classroom_image_analyzer import ClassroomImageAnalyzer
from app.services.llm_chain import multimodal_chain
from app.services.context_manager import context_manager
from app.config.system_config import config
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from fastapi.responses import StreamingResponse
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase environment variables are missing!")
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

def get_embedding(text):
    return embedding_model.encode(text).tolist()

router = APIRouter()
image_analyzer = ClassroomImageAnalyzer()


class ChatRequest(BaseModel):
    message: str
    type: str = "text"

class ChatResponse(BaseModel):
    message: str
    timestamp: str
    input_type: str = "text"

class TextRequest(BaseModel):
    query: str

class TextResponse(BaseModel):
    response: str
    input_type: str = "text"

class VoiceResponse(BaseModel):
    response: str
    transcription: str
    input_type: str = "voice"
    session_id: str

class ImageResponse(BaseModel):
    response: str
    analysis: dict
    input_type: str = "image"
    image_id: str

class MultimodalRequest(BaseModel):
    query: str
    voice_transcription: Optional[str] = None
    image_id: Optional[str] = None
    input_type: str = "text"

class MultimodalResponse(BaseModel):
    response: str
    input_type: str
    image_analysis: Optional[bool] = False
    voice_transcription: Optional[bool] = False

class ContextSummaryResponse(BaseModel):
    recent_messages: int
    active_images: int
    voice_sessions: int
    latest_image: bool
    session_active: bool

class ContextClearResponse(BaseModel):
    status: str
    cleared: str

class ConfigResponse(BaseModel):
    retention: dict
    limits: dict
    models: dict
    background_tasks: dict
    allowed_file_types: dict


def store_query_embedding(query: str):
    """Background task to store query embedding"""
    try:
        embedding = get_embedding(query)
        supabase.table("user_queries").insert({
            "query": query,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat()
        }).execute()
        logger.info(f"Stored embedding for query: {query[:50]}...")
    except Exception as e:
        logger.error(f"Error storing query embedding: {e}")

def cleanup_old_embeddings():
    """Background task to cleanup old embeddings based on config"""
    try:
        
        embedding_retention = config.get_retention_timedelta("embedding")
        document_retention = config.get_retention_timedelta("document")
        
        
        embedding_cutoff = datetime.now() - embedding_retention
        document_cutoff = datetime.now() - document_retention
        
        
        supabase.table("user_queries").delete().lt("timestamp", embedding_cutoff.isoformat()).execute()
        
        
        supabase.table("documents").delete().lt("created_at", document_cutoff.isoformat()).execute()
        
        logger.info(f"Cleaned up embeddings older than {embedding_cutoff.date()} and documents older than {document_cutoff.date()}")
    except Exception as e:
        logger.error(f"Error cleaning up old embeddings: {e}")

def sanitize_query(query: str) -> str:
    """Sanitize and normalize incoming queries using config limits"""
    if not query:
        return ""
    
    
    cleaned = query.strip()
    
    cleaned = cleaned.replace('\n', ' ')
    # Remove multiple spaces
    cleaned = ' '.join(cleaned.split())
    # Limit length based on config
    cleaned = cleaned[:config.MAX_QUERY_LENGTH] if len(cleaned) > config.MAX_QUERY_LENGTH else cleaned
    
    return cleaned

def validate_file_upload(file: UploadFile, file_type: str) -> bool:
    """Validate file upload based on config settings"""
    if not file:
        return False
    
  
    if not config.validate_file_type(file.filename, file_type):
        logger.warning(f"Invalid file type: {file.filename} for type {file_type}")
        return False
    
    
    return True

@router.post("/ask_text", response_model=TextResponse)
async def ask_from_text(request: TextRequest, background_tasks: BackgroundTasks):
    """
    Process text-only input with context awareness
    """
    try:
        # Sanitize input
        clean_query = sanitize_query(request.query)
        if not clean_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Received text request: {clean_query[:50]}...")
        
        # Add background task for embedding storage
        background_tasks.add_task(store_query_embedding, clean_query)
        
        # Get latest image context if available
        latest_image_context = context_manager.get_latest_image_context()
        
        # Process with multimodal chain
        response = multimodal_chain.process_multimodal_input(
            query=clean_query,
            image_analysis=latest_image_context,
            input_type="text"
        )
        
        logger.info(f"Sending text response: {response[:100]}...")
        return TextResponse(response=response, input_type="text")
        
    except Exception as e:
        logger.error(f"Error processing text request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask/voice", response_model=VoiceResponse)
async def ask_from_voice(file: UploadFile = File(...)):
    """
    Process voice input: Whisper → Text → LLM Response
    """
    temp_path = "temp_audio.wav"
    session_id = str(uuid.uuid4())
    
    try:
        # Validate file upload
        if not validate_file_upload(file, "audio"):
            raise HTTPException(status_code=400, detail="Invalid audio file type")
        
        logger.info(f"Received voice file: {file.filename}")
        
        # Save the audio file temporarily
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Transcribe audio using Whisper
        transcription = transcribe_audio(temp_path)
        logger.info(f"Transcribed text: {transcription}")
        
        # Sanitize transcription
        clean_transcription = sanitize_query(transcription)
        
        # Store voice context
        context_manager.add_voice_transcription(session_id, clean_transcription)
        
        # Process with multimodal chain
        response = multimodal_chain.process_multimodal_input(
            query=clean_transcription,
            voice_transcription=clean_transcription,
            input_type="voice"
        )
        
        logger.info(f"Sending voice response: {response[:100]}...")
        return VoiceResponse(
            response=response, 
            transcription=clean_transcription,
            input_type="voice",
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing voice request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/ask/image", response_model=ImageResponse)
async def ask_from_image(
    file: UploadFile = File(...),
    query: Optional[str] = Form("Please analyze this image and provide a detailed description.")
):
    """
    Process image input: OpenCV/YOLO/VIT → Analysis → LLM Response
    """
    image_id = str(uuid.uuid4())
    temp_path = None
    
    try:
        # Validate file upload
        if not validate_file_upload(file, "image"):
            raise HTTPException(status_code=400, detail="Invalid image file type")
        
        logger.info(f"Received image file: {file.filename}")
        
        # Sanitize query
        clean_query = sanitize_query(query) if query else "Please analyze this image and provide a detailed description."
        
        # Save the image file temporarily
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
            
        temp_path = f"temp_{image_id}.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Analyze image using comprehensive analyzer
        analysis_result = image_analyzer.analyze_image_comprehensive(temp_path)
        
        if 'error' in analysis_result:
            raise HTTPException(status_code=500, detail=analysis_result['error'])
        
        # Store the analysis in context
        context_manager.add_image_analysis(image_id, analysis_result)
        
        # Process with multimodal chain
        response = multimodal_chain.process_multimodal_input(
            query=clean_query,
            image_analysis=analysis_result,
            input_type="image"
        )
        
        logger.info(f"Sending image response: {response[:100]}...")
        
        return ImageResponse(
            response=response,
            analysis=analysis_result,
            input_type="image",
            image_id=image_id
        )
                
    except Exception as e:
        logger.error(f"Error processing image request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/multimodal", response_model=MultimodalResponse)
async def process_multimodal(request: MultimodalRequest):
    """
    Process multimodal input combining multiple input types
    """
    try:
        # Sanitize query
        clean_query = sanitize_query(request.query)
        if not clean_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Received multimodal request: {request.input_type}")
        
        # Get relevant context based on input type
        image_analysis = None
        voice_transcription = None
        
        if request.image_id:
            image_analysis = context_manager.get_image_analysis(request.image_id)
        
        if request.voice_transcription:
            voice_transcription = sanitize_query(request.voice_transcription)
        
        # Process with multimodal chain
        response = multimodal_chain.process_multimodal_input(
            query=clean_query,
            voice_transcription=voice_transcription,
            image_analysis=image_analysis,
            input_type=request.input_type
        )
        
        logger.info(f"Sending multimodal response: {response[:100]}...")
        
        return MultimodalResponse(
            response=response,
            input_type=request.input_type,
            image_analysis=image_analysis is not None,
            voice_transcription=voice_transcription is not None
        )
        
    except Exception as e:
        logger.error(f"Error processing multimodal request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    General chat endpoint with context awareness
    """
    try:
        # Sanitize message
        clean_message = sanitize_query(request.message)
        if not clean_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"Received chat request: {clean_message[:50]}... (type: {request.type})")
        
        # Get latest context
        latest_image_context = context_manager.get_latest_image_context()
        
        # Process with appropriate input type
        response = multimodal_chain.process_multimodal_input(
            query=clean_message,
            image_analysis=latest_image_context,
            input_type=request.type
        )
        
        # Create response with timestamp
        chat_response = ChatResponse(
            message=response,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            input_type=request.type
        )
        
        logger.info(f"Sending chat response: {chat_response.message[:100]}...")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context/summary", response_model=ContextSummaryResponse)
async def get_context_summary():
    """
    Get current context summary for debugging
    """
    try:
        summary = context_manager.get_context_summary()
        return ContextSummaryResponse(**summary)
    except Exception as e:
        logger.error(f"Error getting context summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/context/clear", response_model=ContextClearResponse)
async def clear_context(context_type: Optional[str] = None):
    """
    Clear specific or all context
    """
    try:
        context_manager.clear_context(context_type)
        return ContextClearResponse(status="success", cleared=context_type or "all")
    except Exception as e:
        logger.error(f"Error clearing context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config", response_model=ConfigResponse)
async def get_system_config():
    """
    Get current system configuration
    """
    try:
        return ConfigResponse(**config.get_config_summary())
    except Exception as e:
        logger.error(f"Error getting system config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/maintenance/cleanup")
async def trigger_cleanup(background_tasks: BackgroundTasks):
    """
    Trigger cleanup of old embeddings (admin endpoint)
    """
    try:
        background_tasks.add_task(cleanup_old_embeddings)
        return {"status": "success", "message": "Cleanup task scheduled"}
    except Exception as e:
        logger.error(f"Error scheduling cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def sse_format(data: str) -> str:
    return f"data: {data}\n\n"

@router.post("/stream_chat")
async def stream_chat(request: ChatRequest):
    """
    Stream LLM response tokens to the client using SSE.
    """
    query = request.message
    input_type = request.type
    # Optionally, add support for voice/image/multimodal
    # For now, just text streaming
    
    async def event_generator():
        # Use retrieval-augmented context
        # (Assume multimodal_chain.llm supports async streaming)
        # Compose context as in process_multimodal_input
        multimodal_chain.store_kb_entry(query, role="user")
        history_context = multimodal_chain._build_history_context()
        retrieval_context = multimodal_chain.build_retrieval_context(query)
        context = f"{history_context}\n{retrieval_context}".strip()
        chain_input = {"context": context, "query": query}
        chain = multimodal_chain.chat_prompt | multimodal_chain.llm
        # Streaming: yield tokens as they are generated
        try:
            async for chunk in chain.astream(chain_input):
                if chunk:
                    yield sse_format(chunk)
                    await asyncio.sleep(0)  # Yield control
            # Store the full response in KB and history
            # (Assume chain.astream yields the full response at the end)
            # Optionally, collect and store the full response here
        except Exception as e:
            yield sse_format(f"[ERROR] {str(e)}")
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
