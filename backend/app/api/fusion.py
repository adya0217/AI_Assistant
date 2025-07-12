import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from pydantic import BaseModel
import logging
from datetime import datetime, timedelta
import uuid
from typing import Optional, List
from app.services.whisper_stt import record_audio, transcribe_audio
from app.services.classroom_image_analyzer import ClassroomImageAnalyzer
from app.services.llm_chain import multimodal_chain
from app.services.context_manager import context_manager, cache_manager
from app.config.system_config import config
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from fastapi.responses import StreamingResponse
import asyncio
import time
from pydub import AudioSegment
import io
from app.config.openvino_config import OpenVINOConfig
from transformers import AutoTokenizer
try:
    from optimum.intel.openvino import OVModelForFeatureExtraction
    openvino_available = True
except ImportError:
    openvino_available = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase environment variables are missing!")
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

if OpenVINOConfig.should_use_openvino() and openvino_available:
    try:
        openvino_cache_path = OpenVINOConfig.get_model_cache_path(config.EMBEDDING_MODEL)
        if os.path.exists(openvino_cache_path) and not OpenVINOConfig.should_export_models():
            logger.info(f"Loading OpenVINO optimized embedding model from cache: {openvino_cache_path}")
            embedding_model = OVModelForFeatureExtraction.from_pretrained(openvino_cache_path)
            tokenizer = AutoTokenizer.from_pretrained(openvino_cache_path)
        else:
            logger.info("Exporting embedding model to OpenVINO format...")
            os.makedirs(openvino_cache_path, exist_ok=True)
            embedding_model = OVModelForFeatureExtraction.from_pretrained(config.EMBEDDING_MODEL, export=True)
            embedding_model.save_pretrained(openvino_cache_path)
            tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
            tokenizer.save_pretrained(openvino_cache_path)
            logger.info(f"OpenVINO embedding model saved to: {openvino_cache_path}")
        logger.info("OpenVINO optimized embedding model loaded successfully")
        def get_embedding(text):
            encoded = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
            embedding = embedding_model(**encoded).squeeze(0)
            return embedding[0].tolist() if hasattr(embedding, 'tolist') else embedding.tolist()
    except Exception as e:
        logger.warning(f"OpenVINO embedding model not available or failed: {e}\nFalling back to SentenceTransformer.")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        def get_embedding(text):
            return embedding_model.encode(text).tolist()
else:
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
    
    if not query:
        return ""
    
    
    cleaned = query.strip()
    
    cleaned = cleaned.replace('\n', ' ')
    
    cleaned = ' '.join(cleaned.split())
    
    cleaned = cleaned[:config.MAX_QUERY_LENGTH] if len(cleaned) > config.MAX_QUERY_LENGTH else cleaned
    
    return cleaned

def validate_file_upload(file: UploadFile, file_type: str) -> bool:
    
    if not file:
        return False
    
  
    if not config.validate_file_type(file.filename, file_type):
        logger.warning(f"Invalid file type: {file.filename} for type {file_type}")
        return False
    
    
    return True

@router.post("/ask_text", response_model=TextResponse)
async def ask_from_text(request: TextRequest, background_tasks: BackgroundTasks, output_lines: int = Query(None), max_tokens: int = Query(None)):
    start_time = time.time()
    logger.info(f"[ask_text] Request received: {request.query[:50]}...")
    try:
        clean_query = sanitize_query(request.query)
        if not clean_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        logger.info(f"[ask_text] Cleaned query: {clean_query[:50]}...")
        cached = cache_manager.get_text(clean_query)
        if cached:
            logger.info("[ask_text] Returning cached response.")
            if output_lines is not None:
                cached = '\n'.join(cached.split('\n')[:output_lines])
            return TextResponse(response=cached, input_type="text")
        
        background_tasks.add_task(store_query_embedding, clean_query)
        latest_image_context = context_manager.get_latest_image_context()
        model_start = time.time()
        logger.info("[ask_text] Calling LLM...")
        response = multimodal_chain.process_multimodal_input(
            query=clean_query,
            image_analysis=latest_image_context,
            input_type="text",
            max_tokens=max_tokens
        )
        model_time = time.time() - model_start
        logger.info(f"[ask_text] LLM response received in {model_time:.2f}s")
        cache_manager.set_text(clean_query, response)
        # Post-process for output_lines
        if output_lines is not None:
            response = '\n'.join(response.split('\n')[:output_lines])
        total_time = time.time() - start_time
        logger.info(f"[ask_text] Sending response in {total_time:.2f}s: {response[:100]}...")
        return TextResponse(response=response, input_type="text")
    except Exception as e:
        logger.error(f"[ask_text] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask/voice", response_model=VoiceResponse)
async def ask_from_voice(file: UploadFile = File(...), output_lines: int = Query(None), max_tokens: int = Query(None)):
    start_time = time.time()
    temp_path = "temp_audio.wav"
    session_id = str(uuid.uuid4())
    logger.info(f"[ask_voice] Request received: {file.filename}, content-type: {file.content_type}")
    try:
        if not validate_file_upload(file, "audio"):
            raise HTTPException(status_code=400, detail="Invalid audio file type")
        contents = await file.read()
        logger.info(f"[ask_voice] Read {len(contents)} bytes of audio data")
        cached_transcription = cache_manager.get_audio(contents)
        if cached_transcription:
            logger.info("[ask_voice] Returning cached transcription.")
            clean_transcription = sanitize_query(cached_transcription)
            context_manager.add_voice_transcription(session_id, clean_transcription)
            # Get cached response for this transcription
            cached_response = cache_manager.get_text(clean_transcription)
            if cached_response:
                logger.info("[ask_voice] Returning cached response for transcription.")
                response = cached_response
            else:
                model_start = time.time()
                logger.info("[ask_voice] Calling LLM for cached transcription...")
                response = multimodal_chain.process_multimodal_input(
                    query=clean_transcription,
                    voice_transcription=clean_transcription,
                    input_type="voice",
                    max_tokens=max_tokens if max_tokens is not None else 60
                )
                cache_manager.set_text(clean_transcription, response)
                model_time = time.time() - model_start
                logger.info(f"[ask_voice] LLM response for cached transcription in {model_time:.2f}s")
            
            if output_lines is not None:
                response = '\n'.join(response.split('\n')[:output_lines])
            return VoiceResponse(
                response=response,
                transcription=clean_transcription,
                input_type="voice",
                session_id=session_id
            )
        try:
            logger.info("[ask_voice] Converting audio to wav format...")
            format_hint = "webm"
            if file.content_type and "audio/" in file.content_type:
                format_hint = file.content_type.split("/")[-1].split(";")[0]
            elif file.filename and "." in file.filename:
                format_hint = file.filename.split(".")[-1]
            logger.info(f"[ask_voice] Using format hint: {format_hint}")
            
            # Try different audio formats if the first one fails
            conversion_success = False
            for attempt_format in [format_hint, "webm", "mp4", "wav"]:
                try:
                    audio = AudioSegment.from_file(
                        io.BytesIO(contents),
                        format=attempt_format,
                        codec="opus" if attempt_format == "webm" else None
                    )
                    audio = audio.set_frame_rate(16000)
                    audio.export(temp_path, format="wav", parameters=["-ac", "1"])
                    logger.info(f"[ask_voice] Successfully converted to wav using format: {attempt_format}")
                    conversion_success = True
                    break
                except Exception as format_error:
                    logger.warning(f"[ask_voice] Failed to convert with format {attempt_format}: {format_error}")
                    continue
            
            if not conversion_success:
                raise Exception("All audio format conversion attempts failed")
                
        except Exception as e:
            logger.error(f"[ask_voice] Audio conversion failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Could not process audio format. Please ensure you're sending a supported audio format. Error: {str(e)}"
            )
        logger.info("[ask_voice] Starting transcription...")
        transcribe_start = time.time()
        transcription = transcribe_audio(temp_path)
        transcribe_time = time.time() - transcribe_start
        logger.info(f"[ask_voice] Transcribed text in {transcribe_time:.2f}s: {transcription[:100]}")
        cache_manager.set_audio(contents, transcription)
        clean_transcription = sanitize_query(transcription)
        logger.info(f"[ask_voice] Cleaned transcription (after sanitize_query): {clean_transcription}")
        context_manager.add_voice_transcription(session_id, clean_transcription)
        model_start = time.time()
        logger.info(f"[ask_voice] Text sent to LLM: {clean_transcription}")
        logger.info("[ask_voice] Calling LLM...")
        response = multimodal_chain.process_multimodal_input(
            query=clean_transcription,
            voice_transcription=clean_transcription,
            input_type="voice",
            max_tokens=max_tokens if max_tokens is not None else 60
        )
        if output_lines is not None:
            response = '\n'.join(response.split('\n')[:output_lines])
        model_time = time.time() - model_start
        total_time = time.time() - start_time
        logger.info(f"[ask_voice] LLM response in {model_time:.2f}s, total time {total_time:.2f}s")
        return VoiceResponse(
            response=response,
            transcription=clean_transcription,
            input_type="voice",
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"[ask_voice] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"[ask_voice] Temp file {temp_path} removed.")

@router.post("/ask/image", response_model=ImageResponse)
async def ask_from_image(
    file: UploadFile = File(...),
    query: Optional[str] = Form("Please analyze this image and provide a detailed description."),
    output_lines: int = Query(None),
    max_tokens: int = Query(None)
):
    start_time = time.time()
    image_id = str(uuid.uuid4())
    temp_path = None
    logger.info(f"[ask_image] Request received: {file.filename}")
    try:
        if not validate_file_upload(file, "image"):
            raise HTTPException(status_code=400, detail="Invalid image file type")
        user_text = query.strip() if query and query.strip() else None
        default_prompt = "Please analyze this image and provide a detailed description."
        # If user_text is not the default, combine both for the LLM
        if user_text and user_text != default_prompt:
            clean_query = f"{default_prompt}\nUser note: {user_text}"
        else:
            clean_query = default_prompt
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
        cached_analysis = cache_manager.get_image(contents)
        if cached_analysis:
            logger.info("[ask_image] Returning cached analysis.")
            # Get cached response for this analysis
            analysis_key = f"image_analysis_{hash(str(cached_analysis))}"
            cached_response = cache_manager.get_text(analysis_key)
            if cached_response:
                logger.info("[ask_image] Returning cached response for analysis.")
                response = cached_response
            else:
                model_start = time.time()
                logger.info("[ask_image] Calling LLM for cached analysis...")
                response = multimodal_chain.process_multimodal_input(
                    query=clean_query,
                    image_analysis=cached_analysis,
                    input_type="image",
                    max_tokens=max_tokens if max_tokens is not None else 60
                )
                cache_manager.set_text(analysis_key, response)
                model_time = time.time() - model_start
                logger.info(f"[ask_image] LLM response for cached analysis in {model_time:.2f}s")
            
            if output_lines is not None:
                response = '\n'.join(response.split('\n')[:output_lines])
            return ImageResponse(
                response=response,
                analysis=cached_analysis,
                input_type="image",
                image_id=image_id
            )
        temp_path = f"temp_{image_id}.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)
        logger.info("[ask_image] File saved, starting analysis...")
        analysis_start = time.time()
        analysis_result = image_analyzer.analyze_image_comprehensive(temp_path, user_text=user_text)
        analysis_time = time.time() - analysis_start
        logger.info(f"[ask_image] Image analysis in {analysis_time:.2f}s")
        if 'error' in analysis_result:
            raise HTTPException(status_code=500, detail=analysis_result['error'])
        cache_manager.set_image(contents, analysis_result)
        context_manager.add_image_analysis(image_id, analysis_result)
        model_start = time.time()
        logger.info("[ask_image] Calling LLM...")
        response = multimodal_chain.process_multimodal_input(
            query=clean_query,
            image_analysis=analysis_result,
            input_type="image",
            max_tokens=max_tokens if max_tokens is not None else 60
        )
        if output_lines is not None:
            response = '\n'.join(response.split('\n')[:output_lines])
        model_time = time.time() - model_start
        total_time = time.time() - start_time
        logger.info(f"[ask_image] LLM response in {model_time:.2f}s, total time {total_time:.2f}s")
        return ImageResponse(
            response=response,
            analysis=analysis_result,
            input_type="image",
            image_id=image_id
        )
    except Exception as e:
        logger.error(f"[ask_image] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"[ask_image] Temp file {temp_path} removed.")

@router.post("/multimodal", response_model=MultimodalResponse)
async def process_multimodal(request: MultimodalRequest):
    start_time = time.time()
    logger.info(f"[multimodal] Request received: {request.input_type}")
    try:
        clean_query = sanitize_query(request.query)
        if not clean_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        image_analysis = None
        voice_transcription = None
        if request.image_id:
            image_analysis = context_manager.get_image_analysis(request.image_id)
        if request.voice_transcription:
            voice_transcription = sanitize_query(request.voice_transcription)
        model_start = time.time()
        logger.info("[multimodal] Calling LLM...")
        response = multimodal_chain.process_multimodal_input(
            query=clean_query,
            voice_transcription=voice_transcription,
            image_analysis=image_analysis,
            input_type=request.input_type
        )
        model_time = time.time() - model_start
        total_time = time.time() - start_time
        logger.info(f"[multimodal] LLM response in {model_time:.2f}s, total time {total_time:.2f}s")
        return MultimodalResponse(
            response=response,
            input_type=request.input_type,
            image_analysis=image_analysis is not None,
            voice_transcription=voice_transcription is not None
        )
    except Exception as e:
        logger.error(f"[multimodal] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    logger.info(f"[chat] Request received: {request.message[:50]}... (type: {request.type})")
    try:
        clean_message = sanitize_query(request.message)
        if not clean_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        latest_image_context = context_manager.get_latest_image_context()
        model_start = time.time()
        logger.info("[chat] Calling LLM...")
        response = multimodal_chain.process_multimodal_input(
            query=clean_message,
            image_analysis=latest_image_context,
            input_type=request.type
        )
        model_time = time.time() - model_start
        total_time = time.time() - start_time
        logger.info(f"[chat] LLM response in {model_time:.2f}s, total time {total_time:.2f}s")
        chat_response = ChatResponse(
            message=response,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            input_type=request.type
        )
        logger.info(f"[chat] Sending response: {chat_response.message[:100]}...")
        return chat_response
    except Exception as e:
        logger.error(f"[chat] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context/summary", response_model=ContextSummaryResponse)
async def get_context_summary():
   
    try:
        summary = context_manager.get_context_summary()
        return ContextSummaryResponse(**summary)
    except Exception as e:
        logger.error(f"Error getting context summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/context/clear", response_model=ContextClearResponse)
async def clear_context(context_type: Optional[str] = None):
    
    try:
        context_manager.clear_context(context_type)
        return ContextClearResponse(status="success", cleared=context_type or "all")
    except Exception as e:
        logger.error(f"Error clearing context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config", response_model=ConfigResponse)
async def get_system_config():
   
    try:
        return ConfigResponse(**config.get_config_summary())
    except Exception as e:
        logger.error(f"Error getting system config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/maintenance/cleanup")
async def trigger_cleanup(background_tasks: BackgroundTasks):
   
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
    
    query = request.message
    input_type = request.type
    
    async def event_generator():
       
        multimodal_chain.store_kb_entry(query, role="user")
        history_context = multimodal_chain._build_history_context()
        retrieval_context = multimodal_chain.build_retrieval_context(query)
        context = f"{history_context}\n{retrieval_context}".strip()
        chain_input = {"context": context, "query": query}
        chain = multimodal_chain.chat_prompt | multimodal_chain.llm
        
        try:
            async for chunk in chain.astream(chain_input):
                if chunk:
                    yield sse_format(chunk)
                    await asyncio.sleep(0)  
            
        except Exception as e:
            yield sse_format(f"[ERROR] {str(e)}")
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
