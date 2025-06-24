"""
Configuration settings for the classroom image analysis system
"""

import os
from typing import Dict, Any
import torch

# Model configurations
MODELS_TO_DOWNLOAD = {
    "YOLO": {
        "name": "yolov8n.pt",
        "description": "YOLOv8 nano model for object detection",
        "auto_download": True
    },
    "OCR": {
        "name": "microsoft/trocr-base-printed",
        "description": "TrOCR model for text extraction",
        "auto_download": True
    },
    "Image_Captioning": {
        "name": "Salesforce/blip-image-captioning-base",
        "description": "BLIP model for image captioning",
        "auto_download": True
    },
    "Visual_QA": {
        "name": "dandelin/vilt-b32-finetuned-vqa",
        "description": "ViLT model for visual question answering",
        "auto_download": True
    }
}

# Hardware recommendations
HARDWARE_RECOMMENDATIONS = {
    "Minimum": {
        "RAM": "8GB",
        "GPU": "CPU only",
        "Notes": "Processing will be slower, consider using smaller models"
    },
    "Recommended": {
        "RAM": "16GB",
        "GPU": "NVIDIA GPU with 6GB+ VRAM",
        "Notes": "Good balance of performance and resource usage"
    },
    "Optimal": {
        "RAM": "32GB",
        "GPU": "NVIDIA RTX 3080+ or Intel Arc GPU",
        "Notes": "Best performance for real-time processing"
    }
}

# Optimization settings
OPTIMIZATION_TIPS = {
    "For_Intel_Hardware": {
        "recommendation": "Use Intel OpenVINO for acceleration",
        "implementation": "Install openvino-dev and convert models"
    },
    "For_Real_time_Processing": {
        "recommendation": "Consider quantized models",
        "implementation": "Use torch.quantization for model optimization"
    },
    "For_Memory_Constraints": {
        "recommendation": "Load models on-demand",
        "implementation": "Use lazy loading in ClassroomImageAnalyzer"
    },
    "For_Accuracy": {
        "recommendation": "Use larger YOLO models",
        "options": ["yolov8m.pt", "yolov8l.pt"],
        "implementation": "Change model path in analyzer initialization"
    }
}

# Model paths and cache settings
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Performance settings
PERFORMANCE_SETTINGS = {
    "batch_size": 1,  # Adjust based on available memory
    "num_workers": 4,  # Adjust based on CPU cores
    "use_half_precision": True if DEVICE == "cuda" else False,
    "cache_size": 100  # Number of images to cache in memory
}

# Subject-specific settings
SUBJECT_SETTINGS = {
    "mathematics": {
        "confidence_threshold": 0.6,
        "keywords": ["equation", "graph", "formula", "geometric", "algebra", "calculus"]
    },
    "science": {
        "confidence_threshold": 0.6,
        "keywords": ["experiment", "laboratory", "chemical", "biology", "physics"]
    },
    "geography": {
        "confidence_threshold": 0.6,
        "keywords": ["map", "continent", "country", "river", "mountain"]
    },
    "history": {
        "confidence_threshold": 0.6,
        "keywords": ["historical", "ancient", "timeline", "monument", "artifact"]
    }
}

def get_device_info() -> Dict[str, Any]:
    """Get information about the current device configuration"""
    return {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB" if torch.cuda.is_available() else "N/A"
    }

def get_optimization_recommendations() -> Dict[str, str]:
    """Get optimization recommendations based on current hardware"""
    recommendations = {}
    
    if not torch.cuda.is_available():
        recommendations["hardware"] = "Consider using GPU for better performance"
    
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:
        recommendations["memory"] = "Consider using smaller models or model quantization"
    
    return recommendations 