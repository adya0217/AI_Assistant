import os
import uuid
import fitz  
import logging
import time
from typing import Dict
from fastapi import APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer


load_dotenv()


router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

@router.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)) -> Dict:
    
    #Uploads a PDF, extracts text, chunks it, embeds, and stores embeddings in Supabase.
    
    start_time = time.time()
    file_ext = os.path.splitext(file.filename)[1]
    logger.info(f"[upload_pdf] Request received: {file.filename}")
    if file_ext.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        logger.info(f"Receiving file: {file.filename}")
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info("[upload_pdf] File saved, extracting text...")
        extract_start = time.time()
        text = extract_text_from_pdf(file_path)
        extract_time = time.time() - extract_start
        logger.info(f"[upload_pdf] Text extracted in {extract_time:.2f}s")

        logger.info("[upload_pdf] Chunking extracted text...")
        chunk_start = time.time()
        chunks = chunk_text(text)
        chunk_time = time.time() - chunk_start
        logger.info(f"[upload_pdf] Text chunked in {chunk_time:.2f}s")

        logger.info("[upload_pdf] Embedding and uploading chunks...")
        embed_start = time.time()
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            response = supabase.table("documents").insert({
                "content": chunk,
                "embedding": embedding
            }).execute()
            if response.get("status_code") not in [200, 201]:
                raise Exception(f"Failed to insert chunk {i+1} into Supabase.")
        embed_time = time.time() - embed_start
        logger.info(f"[upload_pdf] Embedded and uploaded {len(chunks)} chunks in {embed_time:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"[upload_pdf] Successfully uploaded {len(chunks)} chunks. Total time: {total_time:.2f}s")
        return {"status": "success", "chunks_uploaded": len(chunks)}

    except Exception as e:
        logger.error(f"[upload_pdf] Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("[upload_pdf] Temporary file cleaned up.")
