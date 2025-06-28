import torch
from torchvision import models
import torch.nn as nn
import os
from torchvision import transforms

def load_vit_model(weights_path):
    """Load the trained ViT model for landmark/historical classification"""
    try:
        model = models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        state_dict.pop('heads.head.weight', None)
        state_dict.pop('heads.head.bias', None)
        
        model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        
        class_names = ['landmark', 'historical']
        
        print(f"ViT model loaded from {weights_path}. Feature extractor weights loaded, but the final classification layer was re-initialized for 2 classes due to a size mismatch.")
        return model, class_names
        
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return None, []

def predict_vit_class(image, model, class_names, topk=3):
    """Predict class using ViT model"""
    try:
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
            
        return results
        
    except Exception as e:
        print(f"Error in ViT prediction: {e}")
        return None 