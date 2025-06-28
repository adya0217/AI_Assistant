import os
from typing import Dict, Any
from datetime import timedelta

class SystemConfig:
  
    
    
    EMBEDDING_RETENTION_DAYS = int(os.getenv("EMBEDDING_RETENTION_DAYS", "30"))
    DOCUMENT_RETENTION_DAYS = int(os.getenv("DOCUMENT_RETENTION_DAYS", "90"))
    CONTEXT_HISTORY_LIMIT = int(os.getenv("CONTEXT_HISTORY_LIMIT", "50"))
    
   
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))
    
    
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    
    LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
    STT_MODEL = os.getenv("STT_MODEL", "base")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    
    BACKGROUND_TASK_TIMEOUT = int(os.getenv("BACKGROUND_TASK_TIMEOUT", "300"))
    CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
    
    
    ALLOWED_FILE_TYPES = {
        "image": [".jpg", ".jpeg", ".png", ".webp", ".gif"],
        "audio": [".wav", ".mp3", ".m4a", ".webm"],
        "document": [".pdf", ".txt", ".doc", ".docx"]
    }
    
    
    SUBJECT_KEYWORDS = {
        'mathematics': [
            'equation', 'graph', 'formula', 'geometric', 'algebra', 'calculus',
            'triangle', 'circle', 'square', 'angle', 'coordinate', 'function',
            'derivative', 'integral', 'matrix', 'vector', 'probability'
        ],
        'science': [
            'experiment', 'laboratory', 'chemical', 'biology', 'physics',
            'molecule', 'atom', 'cell', 'microscope', 'circuit', 'magnet',
            'plant', 'animal', 'reaction', 'force', 'energy', 'wave'
        ],
        'geography': [
            'map', 'continent', 'country', 'river', 'mountain', 'climate',
            'population', 'capital', 'ocean', 'latitude', 'longitude'
        ],
        'history': [
            'historical', 'ancient', 'timeline', 'monument', 'artifact',
            'civilization', 'empire', 'war', 'revolution', 'culture',
            'taj mahal', 'pyramid', 'colosseum', 'eiffel tower', 'great wall',
            'statue of liberty', 'temple', 'palace', 'fort', 'mosque',
            'cathedral', 'tomb', 'mausoleum', 'architectural', 'heritage'
        ],
        'language': [
            'text', 'paragraph', 'sentence', 'grammar', 'vocabulary',
            'literature', 'poem', 'story', 'essay', 'letter'
        ]
    }
    
    @classmethod
    def get_retention_timedelta(cls, data_type: str = "embedding") -> timedelta:
        """Get retention period for different data types"""
        if data_type == "document":
            return timedelta(days=cls.DOCUMENT_RETENTION_DAYS)
        else:
            return timedelta(days=cls.EMBEDDING_RETENTION_DAYS)
    
    @classmethod
    def validate_file_type(cls, filename: str, file_type: str) -> bool:
        """Validate if file type is allowed"""
        if file_type not in cls.ALLOWED_FILE_TYPES:
            return False
        
        file_extension = os.path.splitext(filename.lower())[1]
        return file_extension in cls.ALLOWED_FILE_TYPES[file_type]
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "retention": {
                "embedding_days": cls.EMBEDDING_RETENTION_DAYS,
                "document_days": cls.DOCUMENT_RETENTION_DAYS,
                "context_history_limit": cls.CONTEXT_HISTORY_LIMIT
            },
            "limits": {
                "max_query_length": cls.MAX_QUERY_LENGTH,
                "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
                "max_image_dimension": cls.MAX_IMAGE_DIMENSION,
                "max_concurrent_requests": cls.MAX_CONCURRENT_REQUESTS
            },
            "models": {
                "llm": cls.LLM_MODEL,
                "stt": cls.STT_MODEL,
                "embedding": cls.EMBEDDING_MODEL
            },
            "background_tasks": {
                "timeout_seconds": cls.BACKGROUND_TASK_TIMEOUT,
                "cleanup_interval_hours": cls.CLEANUP_INTERVAL_HOURS
            },
            "allowed_file_types": cls.ALLOWED_FILE_TYPES
        }

config = SystemConfig() 