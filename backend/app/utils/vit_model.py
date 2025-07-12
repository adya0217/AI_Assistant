import torch
from torchvision import models
import torch.nn as nn
import os
from torchvision import transforms
from app.config.openvino_config import OpenVINOConfig
import logging

logger = logging.getLogger("vit_model")
logging.basicConfig(level=logging.INFO)

def load_vit_model(weights_path):
    try:
        # Force CPU for ViT model due to limited GPU memory
        device = torch.device('cpu')
        if OpenVINOConfig.should_use_openvino():
            try:
                from optimum.intel import OVModelForImageClassification
                from transformers import AutoImageProcessor
                
                # Check if OpenVINO model already exists
                openvino_cache_path = OpenVINOConfig.get_model_cache_path("vit_b_16")
                if os.path.exists(openvino_cache_path) and not OpenVINOConfig.should_export_models():
                    logger.info(f"Loading OpenVINO optimized ViT from cache: {openvino_cache_path}")
                    model = OVModelForImageClassification.from_pretrained(openvino_cache_path)
                    processor = AutoImageProcessor.from_pretrained(openvino_cache_path)
                else:
                    logger.info("Exporting ViT to OpenVINO format...")
                    # Load original model first
                    original_model = models.vit_b_16(pretrained=False)
                    original_model.heads.head = nn.Linear(original_model.heads.head.in_features, 2)
                    
                    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
                    state_dict.pop('heads.head.weight', None)
                    state_dict.pop('heads.head.bias', None)
                    original_model.load_state_dict(state_dict, strict=False)
                    original_model.eval()
                    
                    # Export to OpenVINO format
                    os.makedirs(openvino_cache_path, exist_ok=True)
                    model = OVModelForImageClassification.from_pretrained(
                        "google/vit-base-patch16-224",
                        export=True,
                        task="image-classification"
                    )
                    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                    
                    # Save the OpenVINO model
                    model.save_pretrained(openvino_cache_path)
                    processor.save_pretrained(openvino_cache_path)
                    logger.info(f"OpenVINO ViT model saved to: {openvino_cache_path}")
                
                # Load class names from dataset or use defaults
                class_names = _load_class_names_from_dataset()
                logger.info(f"OpenVINO optimized ViT model loaded successfully")
                return model, class_names, processor
                
            except ImportError:
                logger.warning("OpenVINO not available, falling back to PyTorch model")
                return _load_pytorch_vit(weights_path)
        else:
            return _load_pytorch_vit(weights_path)
            
    except Exception as e:
        logger.error(f"Error loading ViT model: {e}")
        return None, [], None

def _load_class_names_from_dataset():
    """Load class names from dataset directories"""
    try:
        # Get the path to dataset directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        landmark_dir = os.path.join(current_dir, '../../../dataset/landmark')
        history_dir = os.path.join(current_dir, '../../../dataset/History')
        
        class_names = set()
        
        for base_dir in [landmark_dir, history_dir]:
            if os.path.isdir(base_dir):
                for entry in os.listdir(base_dir):
                    entry_path = os.path.join(base_dir, entry)
                    if os.path.isdir(entry_path):
                        class_names.add(entry)
        
        if class_names:
            return sorted(list(class_names))
        else:
            print("No class names found in dataset directories, using defaults")
            return ['landmark', 'historical']
            
    except Exception as e:
        print(f"Error loading class names from dataset: {e}")
        return ['landmark', 'historical']

def _load_pytorch_vit(weights_path):
    """Load PyTorch ViT model (fallback)"""
    try:
        # Load class names first
        class_names = _load_class_names_from_dataset()
        
        # Load checkpoint to determine number of classes
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        # Check if checkpoint has class names saved
        if 'class_names' in state_dict:
            checkpoint_classes = state_dict['class_names']
            print(f"Found class names in checkpoint: {checkpoint_classes}")
            class_names = checkpoint_classes
        
        # Determine number of classes
        if 'heads.head.weight' in state_dict:
            num_classes = state_dict['heads.head.weight'].shape[0]
            print(f"Detected num_classes in checkpoint: {num_classes}")
            
            # Adjust class names if needed
            if num_classes != len(class_names):
                print(f"Warning: Checkpoint has {num_classes} classes but dataset has {len(class_names)} classes")
                if num_classes > len(class_names):
                    # Pad class names if needed
                    while len(class_names) < num_classes:
                        class_names.append(f"class_{len(class_names)}")
                else:
                    # Truncate class names if needed
                    class_names = class_names[:num_classes]
        else:
            num_classes = len(class_names)
            print(f"Using dataset number of classes: {num_classes}")
        
        # Create model with correct number of classes
        model = models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        model.to(torch.device('cpu'))
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"PyTorch ViT model loaded from {weights_path} with {num_classes} classes: {class_names}")
        return model, class_names, None
        
    except Exception as e:
        print(f"Error loading PyTorch ViT model: {e}")
        # Fallback to default model
        model = models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        model.eval()
        return model, ['landmark', 'historical'], None

def predict_vit_class(image, model, class_names, processor=None, topk=3):
    import time
    try:
        start_time = time.time()
        if processor is not None:
            # OpenVINO model with processor
            if hasattr(image, 'convert'):
                inputs = processor(image, return_tensors="pt")
            else:
                # Convert numpy array to PIL if needed
                from PIL import Image
                if isinstance(image, torch.Tensor):
                    image = transforms.ToPILImage()(image)
                inputs = processor(image, return_tensors="pt")
            
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = probs.topk(topk)
            results = [(class_names[idx], float(prob)) for idx, prob in zip(top_idxs[0], top_probs[0])]
            
        else:
            # PyTorch model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            if hasattr(image, 'convert'):
                image_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
            else:
                image_tensor = image.unsqueeze(0) if image.dim() == 3 else image
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                top_probs, top_idxs = probs.topk(topk)
                results = [(class_names[idx], float(prob)) for idx, prob in zip(top_idxs[0], top_probs[0])]
            
        inference_time = time.time() - start_time
        logger.info(f"ViT prediction completed in {inference_time:.3f}s")
        return results
        
    except Exception as e:
        logger.error(f"Error in ViT prediction: {e}")
        return None 