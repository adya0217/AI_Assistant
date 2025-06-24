import torch
from torchvision import models
import torch.nn as nn
import os
from torchvision import transforms

def load_vit_model(weights_path):
    """Load the trained ViT model for landmark/historical classification"""
    try:
        # Create ViT model with the architecture you want to use (2 classes)
        model = models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)  # 2 classes: landmark, historical
        
        # Load the weights from your checkpoint file.
        # The error log indicates the checkpoint has a head with 3 classes.
        # We will load all weights *except* for the final classification head to avoid a size mismatch error.
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        # Remove the incompatible head from the loaded weights
        state_dict.pop('heads.head.weight', None)
        state_dict.pop('heads.head.bias', None)
        
        # Load the remaining weights (the feature extractor part) into your new model.
        # `strict=False` allows loading the weights that match, and ignores the ones that don't (like our new head).
        model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        
        # Define class names (should match your training)
        class_names = ['landmark', 'historical']
        
        print(f"ViT model loaded from {weights_path}. Feature extractor weights loaded, but the final classification layer was re-initialized for 2 classes due to a size mismatch.")
        return model, class_names
        
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return None, []

def predict_vit_class(image, model, class_names, topk=3):
    """Predict class using ViT model"""
    try:
        # Define the same transforms used during training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Preprocess image
        if hasattr(image, 'convert'):
            # If it's a PIL Image
            image_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
        else:
            # If it's already a tensor
            image_tensor = image.unsqueeze(0) if image.dim() == 3 else image
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_idxs = probs.topk(topk)
            results = [(class_names[idx], float(prob)) for idx, prob in zip(top_idxs[0], top_probs[0])]
            
        return results
        
    except Exception as e:
        print(f"Error in ViT prediction: {e}")
        return None 