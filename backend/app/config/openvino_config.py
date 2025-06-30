import os
from pathlib import Path
from transformers import AutoConfig

class OpenVINOConfig:
    """Configuration for OpenVINO optimization"""
    
    OPENVINO_MODEL_DIR = os.getenv("OPENVINO_MODEL_DIR", "openvino_models")
    
    # Model names
    WHISPER_MODEL_NAME = "openai/whisper-base"
    BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    VIT_MODEL_NAME = "google/vit-base-patch16-224"
    TROCR_MODEL_NAME = "microsoft/trocr-base-printed"
    
    # OpenVINO settings
    OPENVINO_DEVICE = "CPU"  # or "GPU" if available
    OPENVINO_CACHE_DIR = "openvino_models"
    
    # Enable OpenVINO for Whisper by default, disable for others due to NumPy compatibility
    USE_OPENVINO = os.getenv("USE_OPENVINO", "true").lower() == "true"
    
    @classmethod
    def should_use_openvino(cls):
        """Check if OpenVINO should be used"""
        return cls.USE_OPENVINO
    
    @classmethod
    def should_export_models(cls):
        """Check if models should be exported to OpenVINO format"""
        return os.getenv("EXPORT_MODELS", "true").lower() == "true"
    
    @classmethod
    def get_model_cache_path(cls, model_name):
        """Get the path where OpenVINO model should be cached"""
        base_dir = os.path.join(os.getcwd(), cls.OPENVINO_MODEL_DIR)
        return os.path.join(base_dir, model_name)

    @classmethod
    def is_encoder_decoder_model(cls, model_name):
        """Check if a model is an encoder-decoder architecture"""
        try:
            config = AutoConfig.from_pretrained(model_name)
            return getattr(config, "is_encoder_decoder", False)
        except Exception as e:
            print(f"Error checking model architecture: {e}")
            return False

    @classmethod
    def get_model_type(cls, model_name):
        """Get the type of model for proper handling"""
        if model_name == cls.WHISPER_MODEL_NAME:
            return "speech"
        elif model_name == cls.TROCR_MODEL_NAME:
            return "vision-encoder-decoder"
        elif model_name == cls.VIT_MODEL_NAME:
            return "vision"
        elif model_name == cls.BLIP_MODEL_NAME:
            return "vision-encoder-decoder"
        else:
            return "unknown"

    @classmethod
    def get_model_class(cls, model_type):
        """Get the appropriate model class based on model type"""
        from optimum.intel.openvino import (
            OVModelForSpeechSeq2Seq,
            OVModelForVision2Seq,
            OVModelForImageClassification
        )
        
        model_classes = {
            "speech": OVModelForSpeechSeq2Seq,
            "vision-encoder-decoder": OVModelForVision2Seq,
            "vision": OVModelForImageClassification
        }
        return model_classes.get(model_type)

    @classmethod
    def export_model(cls, model_name, cache_path):
        """Export model to OpenVINO format with proper configuration"""
        model_type = cls.get_model_type(model_name)
        model_class = cls.get_model_class(model_type)
        
        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        print(f"Exporting {model_name} as {model_type} to OpenVINO format...")
        model = model_class.from_pretrained(model_name, export=True)
        model.save_pretrained(cache_path)
        return model 