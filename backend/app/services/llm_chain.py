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

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

class MultimodalChain:
    """
    Comprehensive multimodal chaining system that combines:
    - Voice: Whisper → Text
    - Image: OpenCV/YOLO/VIT → Description/Text  
    - Text Input: Normal chat
    - Context Manager: ChatGPT-style memory
    """
    
    def __init__(self):
        self.llm = OllamaLLM(model="mistral")
        self.setup_prompts()
        self.k_retrieval = 4
        
    def setup_prompts(self):
        """Setup different prompt templates for various input types"""
        
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
        return embedding_model.encode(text).tolist()

    def store_kb_entry(self, text: str, role: str = "user"):
        """Store a query or response in the KB with embedding"""
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
        """Store an image embedding in the database."""
        try:
            supabase.table("image_embeddings").insert({
                "image_path": image_path,
                "embedding": embedding,
            }).execute()
            logger.info(f"Stored image embedding for: {image_path}")
        except Exception as e:
            logger.error(f"Error storing image embedding: {e}")

    def retrieve_similar_images(self, embedding: List[float], k: int = 3, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Retrieve top-k similar images from Supabase."""
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
        """Retrieve top-k similar entries from Supabase using a dedicated RPC function."""
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
        """Retrieve similar queries/responses and build context string."""
        retrieved = self.retrieve_similar(query)
        if not retrieved:
            return ""
        context = "\n---\nRetrieved Knowledge:\n"
        for item in retrieved:
            if item.get("similarity", 0) > 0.75:
                role = item.get("role", "user")
                text = item.get("query", "")
                context += f"[{role}] {text}\n"
        return context

    def process_multimodal_input(
        self, 
        query: str, 
        voice_transcription: Optional[str] = None,
        image_analysis: Optional[Dict[str, Any]] = None,
        input_type: str = "text"
    ) -> str:
        """
        Process multimodal input, retrieve context, and generate response
        """
        try:
            logger.info(f"Processing {input_type} input: {query}")
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
            
            chain = prompt | self.llm
            logger.info(f"Running {input_type} chain with retrieval context")
            response = chain.invoke(chain_input)
            
            self.store_kb_entry(response, role="assistant")
            context_manager.add_to_history(query, response)
            logger.info(f"Generated response: {response[:100]}...")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in multimodal processing: {str(e)}", exc_info=True)
            raise Exception(f"Failed to process {input_type} input: {str(e)}")

    def _build_history_context(self) -> str:
        """Build context from conversation history"""
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
        """Build analysis of current input for multimodal context"""
        analysis = []
        
        if voice_transcription:
            analysis.append(f"Voice Input: {voice_transcription}")
            
        if image_analysis:
            analysis.append(f"Image Analysis: {self._format_image_analysis(image_analysis)}")
            
        if query:
            analysis.append(f"Text Query: {query}")
            
        return "\n".join(analysis) if analysis else "No specific input analysis available"

    def _format_image_analysis(self, image_analysis: Optional[Dict[str, Any]]) -> str:
        """Format image analysis results for prompt inclusion, focusing on the final prediction."""
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
    """Legacy function for backward compatibility"""
    if image_context:
        image_analysis = {"explanation": image_context}
        return multimodal_chain.process_multimodal_input(
            query, image_analysis=image_analysis, input_type="image"
        )
    else:
        return multimodal_chain.process_multimodal_input(query, input_type="text")
