import cv2
import numpy as np
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    TrOCRProcessor, VisionEncoderDecoderModel,
    pipeline
)
from ultralytics import YOLO
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import re
from typing import Dict, List, Tuple, Any
import logging
import os
from app.utils.vit_model import load_vit_model, predict_vit_class
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import vit_b_16
from PIL import Image

BLIP_LOCAL_PATH = os.getenv("BLIP_LOCAL_PATH", "blip-image-captioning-base")

class ClassroomImageAnalyzer:
    """
    Comprehensive image analysis system for classroom AI assistant
    Handles multiple subjects from primary to 10th grade
    """
    
    def __init__(self):
        self.setup_logging()
        self.load_models()
        self.subject_keywords = self.initialize_subject_mapping()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load all required models for comprehensive image analysis"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/yolo/best.pt'))
            print("Loading YOLO from:", yolo_path)
            self.yolo_model = YOLO(yolo_path)
            
            self.ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed', use_fast=True)
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            
            self.blip_processor = BlipProcessor.from_pretrained(BLIP_LOCAL_PATH, use_fast=True)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_LOCAL_PATH)
            
            self.vqa_pipeline = pipeline("visual-question-answering", 
                                       model="dandelin/vilt-b32-finetuned-vqa")

            vit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/vit/vit_finetuned.pth'))
            print("Loading ViT for classification and embeddings from:", vit_path)
            state_dict = torch.load(vit_path, map_location=self.device)
            num_classes = state_dict['heads.head.weight'].shape[0]
            print(f"Detected num_classes in checkpoint: {num_classes}")

            landmark_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/landmark'))
            history_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/History'))
            class_names = set()
            for base_dir in [landmark_dir, history_dir]:
                if os.path.isdir(base_dir):
                    for entry in os.listdir(base_dir):
                        entry_path = os.path.join(base_dir, entry)
                        if os.path.isdir(entry_path):
                            class_names.add(entry)
            self.vit_classes = sorted(class_names)
            print(f"Loaded {len(self.vit_classes)} class names for ViT: {self.vit_classes}")

            self.vit_model = vit_b_16(weights=None)
            self.vit_model.heads.head = torch.nn.Linear(self.vit_model.heads.head.in_features, num_classes)
            self.vit_model.load_state_dict(state_dict)
            self.vit_model.to(self.device)
            self.vit_model.eval()

            self.embedding_model = vit_b_16(weights=None)
            self.embedding_model.heads.head = torch.nn.Linear(self.embedding_model.heads.head.in_features, num_classes)
            self.embedding_model.load_state_dict(state_dict)
            self.embedding_model.heads.head = torch.nn.Identity()
            self.embedding_model.to(self.device)
            self.embedding_model.eval()

            self.embedding_transform = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
            
    def initialize_subject_mapping(self) -> Dict[str, List[str]]:
        """Map subjects to relevant keywords for better context understanding"""
        return {
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
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        self.logger.info(f"[PREPROCESS] Loading image from: {image_path}")
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"[PREPROCESS] Could not load image from {image_path}")
                raise ValueError(f"Could not load image from {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.logger.info(f"[PREPROCESS] Image shape: {image_rgb.shape}")
            return image_rgb
        except Exception as e:
            self.logger.error(f"[PREPROCESS] Error: {e}")
            return None
    
    def detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            self.logger.info("[YOLO] Starting object detection")
            results = self.yolo_model(image)
            detections = []
            for result in results:
                boxes = result.boxes
                self.logger.info(f"[YOLO] Found {len(boxes) if boxes is not None else 0} objects")
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        confidence = float(box.conf[0])
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()
                        }
                        detections.append(detection)
                        self.logger.info(f"[YOLO] Detected {class_name} with confidence {confidence:.2f}")
            self.logger.info(f"[YOLO] All detections: {detections}")
            return {'detections': detections}
        except Exception as e:
            self.logger.error(f"[YOLO] Error: {e}")
            return {'detections': []}
    
    def extract_text_ocr(self, image: np.ndarray) -> str:
        self.logger.info("[OCR] Starting extraction")
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
            pixel_values = self.ocr_processor(pil_image, return_tensors="pt").pixel_values
            generated_ids = self.ocr_model.generate(pixel_values)
            generated_text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.logger.info(f"[OCR] Extracted text: {generated_text.strip()}")
            return generated_text.strip()
        except Exception as e:
            self.logger.error(f"[OCR] Error: {e}")
            return ""
    
    def generate_image_caption(self, image: np.ndarray) -> str:
        self.logger.info("[BLIP] Generating caption")
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
            prompts = ["a photo of", "an image showing"]
            captions = []
            for prompt in prompts:
                inputs = self.blip_processor(pil_image, text=prompt, return_tensors="pt")
                out = self.blip_model.generate(**inputs, max_length=50, num_beams=3)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
            combined_caption = " ".join(set(captions))
            import re
            cleaned_caption = re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', combined_caption)
            cleaned_caption = re.sub(r'(\b\w+\b\s+\b\w+\b)(?:\s+\1)+', r'\1', cleaned_caption)
            self.logger.info(f"[BLIP] Caption: {cleaned_caption.strip()}")
            return cleaned_caption.strip()
        except Exception as e:
            self.logger.error(f"[BLIP] Error: {e}")
            return "Unable to generate caption"
    
    def answer_visual_question(self, image: np.ndarray, question: str) -> str:
        self.logger.info(f"Answering visual question: {question}")
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
            result = self.vqa_pipeline(pil_image, question)
            answer = result[0]['answer'] if result else "Unable to answer"
            self.logger.info(f"Visual QA answer: {answer}")
            return answer
        except Exception as e:
            self.logger.error(f"Error in visual QA: {e}")
            return "Unable to process visual question"
    
    def identify_subject_context(self, text: str, objects: List[str], caption: str) -> str:
        """Identify the most likely subject based on content analysis"""
        combined_text = f"{text} {' '.join(objects)} {caption}".lower()
        
        subject_scores = {}
        for subject, keywords in self.subject_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                subject_scores[subject] = score
        
        if subject_scores:
            return max(subject_scores, key=subject_scores.get)
        return "general"
    
    def generate_educational_explanation(self, image_analysis: Dict[str, Any]) -> str:
        """Generate a general educational explanation for any image."""
        text_content = image_analysis.get('text', '')
        objects = [d['class'] for d in image_analysis.get('objects', [])]
        caption = image_analysis.get('caption', '')
        subject = image_analysis.get('subject', 'general')
        
        if subject == 'history':
            return self.generate_historical_explanation(text_content, objects, caption)
        return self.generate_general_explanation(text_content, objects, caption)

    def generate_general_explanation(self, text: str, objects: List[str], caption: str) -> str:
        """Generate general explanations for non-specific content"""
        explanation = "I've analyzed this image and found the following:\n\n"
        
        if text:
            explanation += f"Text Content: {text}\n"
        
        if objects:
            explanation += f"Objects Detected: {', '.join(set(objects))}\n"
        
        if caption:
            explanation += f"Image Description: {caption}\n"
        
        explanation += "\nEducational Context:\n"
        explanation += "This image appears to be educational content. "
        explanation += "I can help explain specific concepts if you ask me questions about what you see. "
        
        return explanation

    def generate_historical_explanation(self, text: str, objects: List[str], caption: str) -> str:
        """Generate specific explanations for historical monuments and landmarks"""
        explanation = "I've analyzed this historical image and found the following:\n\n"
       
        if text:
            explanation += f"Text Content: {text}\n"
        if objects:
            explanation += f"Objects Detected: {', '.join(set(objects))}\n"
        if caption:
            explanation += f"Image Description: {caption}\n"
        explanation += "\nHistorical Context:\n"
        explanation += "This appears to be a historical monument or landmark. "
        explanation += "I can provide more specific information about its historical significance, architectural features, and cultural importance. "
        explanation += "Would you like to know more about any particular aspect of this historical site?"
        return explanation
    
    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate embedding for a given image using the fine-tuned ViT model."""
        self.logger.info("[ViT Embedding] Starting embedding generation")
        try:
            pil_image = Image.fromarray(image).convert('RGB')
            transformed_image = self.embedding_transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedding_model(transformed_image)
            self.logger.info("[ViT Embedding] Embedding generated successfully")
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            self.logger.error(f"[ViT Embedding] Error: {e}")
            return np.array([])

    def classify_vit(self, image: np.ndarray) -> dict:
        self.logger.info("[ViT] Starting classification")
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
            if self.vit_model and self.vit_classes:
                results = predict_vit_class(pil_image, self.vit_model, self.vit_classes, topk=3)
                self.logger.info(f"[ViT] Top-3 Results: {results}")
                return {"top_k": results}
            else:
                self.logger.error("[ViT] Model or classes not loaded")
                return None
        except Exception as e:
            self.logger.error(f"[ViT] Error: {e}")
            return None
    
    def _get_highest_confidence_yolo(self, detections):
        if not detections:
            return 0.0
        return max(d['confidence'] for d in detections)

    def _get_most_confident_label_yolo(self, detections):
        if not detections:
            return None
        return max(detections, key=lambda d: d['confidence'])['class']

    def analyze_image_comprehensive(self, image_path: str, specific_questions: List[str] = None, user_text: str = None) -> Dict[str, Any]:
        self.logger.info(f"[ANALYZE] Starting comprehensive analysis for: {image_path}")
        image = self.preprocess_image(image_path)
        if image is None:
            self.logger.error("[ANALYZE] Image preprocessing failed")
            return {'error': 'Could not process image'}
        try:
            objects_info = self.detect_objects(image)
            yolo_detections = objects_info['detections']
            yolo_conf = self._get_highest_confidence_yolo(yolo_detections)
            yolo_label = self._get_most_confident_label_yolo(yolo_detections)
            self.logger.info(f"[ANALYZE] YOLO highest confidence: {yolo_conf}, label: {yolo_label}")

            vit_result = self.classify_vit(image)
            vit_label = vit_result['top_k'][0][0] if vit_result else None
            vit_conf = vit_result['top_k'][0][1] if vit_result else 0.0
            self.logger.info(f"[ANALYZE] ViT label: {vit_label}, confidence: {vit_conf}")

            caption = self.generate_image_caption(image)
            self.logger.info(f"[ANALYZE] BLIP caption: {caption}")

            ocr_text = self.extract_text_ocr(image)
            self.logger.info(f"[ANALYZE] OCR text: {ocr_text}")
            text_content = user_text if user_text else ocr_text

            if yolo_detections and yolo_conf > 0.6:
                final_prediction = {
                    'type': 'lab_apparatus',
                    'label': yolo_label,
                    'confidence': yolo_conf
                }
                self.logger.info(f"[ANALYZE] Final prediction: YOLO (lab_apparatus)")
            elif vit_label and vit_conf > 0.6:
                final_prediction = {
                    'type': 'landmark_or_historical',
                    'label': vit_label,
                    'confidence': vit_conf
                }
                self.logger.info(f"[ANALYZE] Final prediction: ViT (landmark_or_historical)")
            else:
                final_prediction = {
                    'type': 'caption_only',
                    'label': caption,
                    'confidence': 0.4
                }
                self.logger.info(f"[ANALYZE] Final prediction: BLIP (caption_only)")

            object_names = [d['class'] for d in yolo_detections]
            subject = self.identify_subject_context(text_content, object_names, caption)
            self.logger.info(f"[ANALYZE] Subject: {subject}")

            qa_results = {}
            if specific_questions:
                for question in specific_questions:
                    answer = self.answer_visual_question(image, question)
                    qa_results[question] = answer
                    self.logger.info(f"[ANALYZE] Visual QA: {question} -> {answer}")

            analysis_result = {
                'objects': yolo_detections,
                'text': text_content,
                'ocr_text': ocr_text,
                'caption': caption,
                'subject': subject,
                'vit_classification': vit_result,
                'qa_results': qa_results,
                'image_path': image_path,
                'user_text': user_text,
                'final_prediction': final_prediction
            }
            analysis_result['explanation'] = self.generate_educational_explanation(analysis_result)
            self.logger.info(f"[ANALYZE] Comprehensive analysis complete for: {image_path}")
            return analysis_result
        except Exception as e:
            self.logger.error(f"[ANALYZE] Error in comprehensive analysis: {e}")
            return {'error': str(e)}

    def generate_unified_explanation(self, classification: Dict[str, str], analysis: Dict[str, Any]) -> str:
        """
        Generates a single, context-aware explanation based on the final classification.
        """
        category = classification.get("category")
        prediction = classification.get("prediction")

        if category == "laboratory_apparatus":
            item_name = prediction.replace('_', ' ')
            description = self.lab_apparatus_info.get(prediction, "a piece of laboratory equipment.")
            return f"This appears to be a **{item_name}**. It is {description}"

        if category == "landmark_historical":
           
            return self.generate_historical_explanation(analysis.get('text', ''), [], "")

       
        object_list = ", ".join(set(analysis.get('objects', [])))
        if object_list:
            return f"This image appears to contain: **{object_list}**. I can provide more details if you ask a specific question."
        else:
            return "I've analyzed the image. What would you like to know about it?" 