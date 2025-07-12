from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage, AIMessage
import logging
from .context_manager import context_manager
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
import numpy as np
from app.services.context_manager import cache_manager
from app.config.openvino_config import OpenVINOConfig
from transformers import AutoTokenizer
try:
    from optimum.intel.openvino import OVModelForFeatureExtraction
    openvino_available = True
except ImportError:
    openvino_available = False

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

if OpenVINOConfig.should_use_openvino() and openvino_available:
    try:
        openvino_cache_path = OpenVINOConfig.get_model_cache_path(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        if os.path.exists(openvino_cache_path) and not OpenVINOConfig.should_export_models():
            logger.info(f"Loading OpenVINO optimized embedding model from cache: {openvino_cache_path}")
            embedding_model = OVModelForFeatureExtraction.from_pretrained(openvino_cache_path)
            tokenizer = AutoTokenizer.from_pretrained(openvino_cache_path)
        else:
            logger.info("Exporting embedding model to OpenVINO format...")
            os.makedirs(openvino_cache_path, exist_ok=True)
            embedding_model = OVModelForFeatureExtraction.from_pretrained(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), export=True)
            embedding_model.save_pretrained(openvino_cache_path)
            tokenizer = AutoTokenizer.from_pretrained(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
            tokenizer.save_pretrained(openvino_cache_path)
            logger.info(f"OpenVINO embedding model saved to: {openvino_cache_path}")
        logger.info("OpenVINO optimized embedding model loaded successfully")
        def embed_text(text: str):
            encoded = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
            embedding = embedding_model(**encoded).squeeze(0)
            return embedding[0].tolist() if hasattr(embedding, 'tolist') else embedding.tolist()
    except Exception as e:
        logger.warning(f"OpenVINO embedding model not available or failed: {e}\nFalling back to SentenceTransformer.")
        embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        def embed_text(text: str):
            return embedding_model.encode(text).tolist()
else:
    embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    def embed_text(text: str):
        return embedding_model.encode(text).tolist()

class MultimodalChain:
    
    def __init__(self):
        self.llm = OllamaLLM(model="mistral")
        self.setup_prompts()
        self.k_retrieval = 2  # Reduced from 4 to 2 for faster retrieval
        
    def setup_prompts(self):
        self.multimodal_prompt = PromptTemplate.from_template(
            """You are an advanced AI assistant with multimodal capabilities. You can process:
- Text input from users
- Transcribed speech from voice input  
- Image analysis results from computer vision models
- Historical conversation context

Your responses should be:
- Educational and informative
- Context-aware based on the conversation history
- Specific to the content being discussed
- Helpful for students and learners

{context}

Current Input Analysis:
{input_analysis}

User Query: {query}

Provide a comprehensive, educational response:"""
        )
        
        self.voice_prompt = PromptTemplate.from_template(
            """You are processing a voice input that has been transcribed to text.
The user said: "{transcription}"

{context}

Please respond naturally as if the user spoke to you directly:"""
        )
        
        self.image_prompt = PromptTemplate.from_template(
            """You are analyzing an image with the following computer vision results:

Image Analysis:
{image_analysis}

{context}

User Question: {query}

Provide a detailed, educational explanation of what you see in the image:"""
        )
        
        self.chat_prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant for educational purposes.

{context}

User: {query}

Assistant:"""
        )

    def embed_text(self, text: str) -> List[float]:
        return embed_text(text)

    def store_kb_entry(self, text: str, role: str = "user"):
        try:
            embedding = self.embed_text(text)
            supabase.table("user_queries").insert({
                "query": text,
                "embedding": embedding,
                "role": role,
                "timestamp": datetime.now().isoformat()
            }).execute()
            logger.info(f"Stored {role} KB entry: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error storing KB entry: {e}. Check if 'user_queries' table exists and has correct RLS policies in Supabase.")

    def store_image_embedding(self, image_path: str, embedding: List[float]):
        try:
            supabase.table("image_embeddings").insert({
                "image_path": image_path,
                "embedding": embedding,
            }).execute()
            logger.info(f"Stored image embedding for: {image_path}")
        except Exception as e:
            logger.error(f"Error storing image embedding: {e}")

    def retrieve_similar_images(self, embedding: List[float], k: int = 3, threshold: float = 0.8) -> List[Dict[str, Any]]:
        try:
            res = supabase.rpc(
                "match_images",
                {
                    "p_query_embedding": embedding,
                    "p_match_count": k,
                    "p_match_threshold": threshold
                }
            ).execute()
            
            if hasattr(res, "data") and res.data:
                logger.info(f"Found {len(res.data)} similar images.")
                return res.data
            return []
        except Exception as e:
            logger.error(f"Error retrieving similar images: {e}")
            return []

    def retrieve_similar(self, text: str, k: int = None) -> List[Dict[str, Any]]:
        if k is None:
            k = self.k_retrieval
        query_embedding = self.embed_text(text)
        
        try:
            res = supabase.rpc(
                "match_queries",
                {"p_query_embedding": query_embedding, "p_match_count": k}
            ).execute()
            
            if hasattr(res, "data") and res.data:
                return res.data
            return []
        except Exception as e:
            logger.error(f"Error retrieving similar from Supabase: {e}. Check if 'match_queries' RPC function exists and is correctly defined in Supabase.")
            return []

    def build_retrieval_context(self, query: str) -> str:
        retrieved = self.retrieve_similar(query)
        if not retrieved:
            return ""
        context = "\n---\nRetrieved Knowledge:\n"
        for item in retrieved:
            if item.get("similarity", 0) > 0.8:  # Increased threshold for better quality
                role = item.get("role", "user")
                text = item.get("query", "")
                context += f"[{role}] {text}\n"
        return context

    def process_multimodal_input(
        self, 
        query: str, 
        voice_transcription: Optional[str] = None,
        image_analysis: Optional[Dict[str, Any]] = None,
        input_type: str = "text",
        max_tokens: Optional[int] = None
    ) -> str:
        try:
            logger.info(f"Processing {input_type} input: {query}")
            
            # Check for exact match in cache first
            exact_match = cache_manager.get_text(query)
            if exact_match:
                logger.info(f"Found exact match for query: {query[:50]}...")
                return exact_match
            
            self.store_kb_entry(query, role="user")
            history_context = self._build_history_context()
            retrieval_context = self.build_retrieval_context(query)
            input_analysis = self._build_input_analysis(
                query, voice_transcription, image_analysis, input_type
            )
            if input_type == "voice":
                prompt = self.voice_prompt
                context = history_context
                chain_input = {
                    "transcription": query,
                    "context": context
                }
            elif input_type == "image":
                prompt = self.image_prompt
                context = history_context
                chain_input = {
                    "image_analysis": self._format_image_analysis(image_analysis),
                    "context": context,
                    "query": query
                }
            elif input_type == "multimodal":
                prompt = self.multimodal_prompt
                context = history_context
                chain_input = {
                    "context": context,
                    "input_analysis": input_analysis,
                    "query": query
                }
            else:
                prompt = self.chat_prompt
                context = history_context
                chain_input = {
                    "context": context,
                    "query": query
                }
            
            context = f"{history_context}\n{retrieval_context}".strip()
            
            llm_kwargs = {}
            if max_tokens is not None:
                llm_kwargs['max_length'] = max_tokens
            else:
                from app.config.system_config import config
                llm_kwargs['max_length'] = config.MAX_RESPONSE_TOKENS
            
            chain = prompt | self.llm
            logger.info(f"Running {input_type} chain with retrieval context")
            response = chain.invoke(chain_input, **llm_kwargs)
            
            self.store_kb_entry(response, role="assistant")
            context_manager.add_to_history(query, response)
            logger.info(f"Generated response: {response[:100]}...")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in multimodal processing: {str(e)}", exc_info=True)
            raise Exception(f"Failed to process {input_type} input: {str(e)}")

    def _build_history_context(self) -> str:
        recent_history = context_manager.get_recent_history(limit=10)
        if not recent_history:
            return ""
        
        context = "Recent conversation context:\n"
        for item in recent_history:
            context += f"User: {item['message']}\n"
            context += f"Assistant: {item['response']}\n\n"
        return context

    def _build_input_analysis(
        self, 
        query: str, 
        voice_transcription: Optional[str] = None,
        image_analysis: Optional[Dict[str, Any]] = None,
        input_type: str = "text"
    ) -> str:
        analysis = []
        
        if voice_transcription:
            analysis.append(f"Voice Input: {voice_transcription}")
            
        if image_analysis:
            analysis.append(f"Image Analysis: {self._format_image_analysis(image_analysis)}")
            
        if query:
            analysis.append(f"Text Query: {query}")
            
        return "\n".join(analysis) if analysis else "No specific input analysis available"

    def _format_image_analysis(self, image_analysis: Optional[Dict[str, Any]]) -> str:
        if not image_analysis:
            return "No image analysis available"
            
        formatted = []
        
        if 'final_prediction' in image_analysis:
            pred = image_analysis['final_prediction']
            pred_type = pred.get('type', 'unknown').replace('_', ' ').title()
            label = pred.get('label', 'N/A')
            confidence = pred.get('confidence', 0)
            formatted.append(f"Primary Identification: The system identified this as a '{label}' ({pred_type}) with a confidence of {confidence:.2f}.")

        if 'caption' in image_analysis and image_analysis['caption']:
            formatted.append(f"General Scene Caption: {image_analysis['caption']}")
            
        if 'text' in image_analysis and image_analysis['text']:
            formatted.append(f"Extracted Text from Image: {image_analysis['text']}")
            
        return "\n".join(formatted) if formatted else "Basic image analysis completed, but no specific objects were identified."

multimodal_chain = MultimodalChain()

def get_llm_response(query: str, image_context: str = None) -> str:
    if image_context:
        image_analysis = {"explanation": image_context}
        return multimodal_chain.process_multimodal_input(
            query, image_analysis=image_analysis, input_type="image"
        )
    else:
        return multimodal_chain.process_multimodal_input(query, input_type="text")

def summarize_all_topics(llm, max_tokens: Optional[int] = None):
    """Generate a summary of all topics/questions covered so far using the LLM."""
    topics = cache_manager.get_all_topics()
    if not topics:
        return "No topics have been covered yet."
    # Prepare a summary prompt
    summary_prompt = "Summarize the following topics and questions covered so far as educational notes:\n"
    for t in topics:
        if t['type'] == 'text':
            summary_prompt += f"- Q: {t['question']}\n  A: {t['answer']}\n"
        elif t['type'] == 'audio':
            summary_prompt += f"- Audio transcription: {t['transcription']}\n"
        elif t['type'] == 'image':
            summary_prompt += f"- Image analysis: {t['analysis']}\n"
    # Use the LLM to generate a summary
    llm_kwargs = {}
    if max_tokens is not None:
        llm_kwargs['max_length'] = max_tokens
    else:
        from app.config.system_config import config
        llm_kwargs['max_length'] = config.MAX_RESPONSE_TOKENS
    summary = llm(summary_prompt, **llm_kwargs)
    return summary
