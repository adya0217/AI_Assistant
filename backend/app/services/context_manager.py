from typing import Dict, Optional, List, Any
import logging
from datetime import datetime
import json
from app.config.system_config import config

logger = logging.getLogger(__name__)

class ContextManager:
    
    def __init__(self):
        self.image_context: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.voice_context: Dict[str, str] = {}
        self.current_session: Dict[str, Any] = {}
        self.max_history_length = config.CONTEXT_HISTORY_LIMIT
        
    def add_image_analysis(self, image_id: str, analysis: Dict[str, Any]):
        self.image_context[image_id] = {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'type': 'image_analysis'
        }
        logger.info(f"Stored image analysis for {image_id}")
    
    def get_image_analysis(self, image_id: str) -> Optional[Dict[str, Any]]:
        return self.image_context.get(image_id, {}).get('analysis')
    
    def get_latest_image_context(self) -> Optional[Dict[str, Any]]:
        if not self.image_context:
            return None
        
        latest_id = max(self.image_context.keys(), 
                       key=lambda k: self.image_context[k]['timestamp'])
        return self.image_context[latest_id]['analysis']
    
    def add_voice_transcription(self, session_id: str, transcription: str):
        self.voice_context[session_id] = {
            'transcription': transcription,
            'timestamp': datetime.now().isoformat(),
            'type': 'voice_input'
        }
        logger.info(f"Stored voice transcription for session {session_id}")
    
    def get_voice_context(self, session_id: str) -> Optional[str]:
        return self.voice_context.get(session_id, {}).get('transcription')
    
    def add_to_history(self, message: str, response: str, input_type: str = "text", 
                      metadata: Optional[Dict[str, Any]] = None):
        history_entry = {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "input_type": input_type,
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(history_entry)
        
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)
        
        logger.info(f"Added {input_type} interaction to conversation history")
    
    def get_recent_history(self, limit: int = 10, input_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if limit <= 0:
            limit = min(10, self.max_history_length)
        
        history = self.conversation_history[-limit:] if limit > 0 else self.conversation_history
        
        if input_type:
            history = [entry for entry in history if entry.get('input_type') == input_type]
        
        return history
    
    def get_context_summary(self) -> Dict[str, Any]:
        return {
            'recent_messages': len(self.conversation_history),
            'active_images': len(self.image_context),
            'voice_sessions': len(self.voice_context),
            'latest_image': self.get_latest_image_context() is not None,
            'session_active': bool(self.current_session),
            'max_history_limit': self.max_history_length
        }
    
    def set_session_context(self, session_data: Dict[str, Any]):
        self.current_session.update(session_data)
        self.current_session['last_updated'] = datetime.now().isoformat()
    
    def get_session_context(self) -> Dict[str, Any]:
        return self.current_session.copy()
    
    def clear_context(self, context_type: Optional[str] = None):
        if context_type == "images":
            self.image_context.clear()
            logger.info("Cleared image context")
        elif context_type == "voice":
            self.voice_context.clear()
            logger.info("Cleared voice context")
        elif context_type == "history":
            self.conversation_history.clear()
            logger.info("Cleared conversation history")
        elif context_type == "session":
            self.current_session.clear()
            logger.info("Cleared session context")
        else:
            self.image_context.clear()
            self.voice_context.clear()
            self.conversation_history.clear()
            self.current_session.clear()
            logger.info("Cleared all context")
    
    def cleanup_old_context(self, days: int = None):
        if days is None:
            days = config.EMBEDDING_RETENTION_DAYS
        
        cutoff_date = datetime.now().isoformat()
        cutoff_timestamp = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        old_image_ids = [
            img_id for img_id, data in self.image_context.items()
            if datetime.fromisoformat(data['timestamp']).timestamp() < cutoff_timestamp
        ]
        for img_id in old_image_ids:
            del self.image_context[img_id]
        
        old_voice_ids = [
            voice_id for voice_id, data in self.voice_context.items()
            if datetime.fromisoformat(data['timestamp']).timestamp() < cutoff_timestamp
        ]
        for voice_id in old_voice_ids:
            del self.voice_context[voice_id]
        
        self.conversation_history = [
            entry for entry in self.conversation_history
            if datetime.fromisoformat(entry['timestamp']).timestamp() >= cutoff_timestamp
        ]
        
        logger.info(f"Cleaned up context older than {days} days")
    
    def export_context(self) -> Dict[str, Any]:
        return {
            'image_context': self.image_context,
            'voice_context': self.voice_context,
            'conversation_history': self.conversation_history,
            'current_session': self.current_session,
            'export_timestamp': datetime.now().isoformat(),
            'config': {
                'max_history_length': self.max_history_length,
                'retention_days': config.EMBEDDING_RETENTION_DAYS
            }
        }
    
    def import_context(self, context_data: Dict[str, Any]):
        self.image_context = context_data.get('image_context', {})
        self.voice_context = context_data.get('voice_context', {})
        self.conversation_history = context_data.get('conversation_history', [])
        self.current_session = context_data.get('current_session', {})
        logger.info("Imported context data")
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_images': len(self.image_context),
            'total_voice_sessions': len(self.voice_context),
            'total_conversations': len(self.conversation_history),
            'history_utilization': f"{len(self.conversation_history)}/{self.max_history_length}",
            'oldest_image': min([data['timestamp'] for data in self.image_context.values()]) if self.image_context else None,
            'newest_image': max([data['timestamp'] for data in self.image_context.values()]) if self.image_context else None,
            'oldest_voice': min([data['timestamp'] for data in self.voice_context.values()]) if self.voice_context else None,
            'newest_voice': max([data['timestamp'] for data in self.voice_context.values()]) if self.voice_context else None,
            'session_active': bool(self.current_session)
        }

context_manager = ContextManager() 