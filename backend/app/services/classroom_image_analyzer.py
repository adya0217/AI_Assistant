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
from app.config.openvino_config import OpenVINOConfig
from app.config.system_config import SystemConfig as config
from torch import serialization
# Add LaTeX-OCR import
try:
    from latex_ocr import LatexOCR
    latex_ocr_available = True
except ImportError:
    latex_ocr_available = False
import pytesseract
from PIL import Image as PILImage

BLIP_LOCAL_PATH = os.getenv("BLIP_LOCAL_PATH", "blip-image-captioning-base")

# Add safe globals for PyTorch 2.6 compatibility
SAFE_GLOBALS = [
    'ultralytics.nn.modules.head.Detect',
    'ultralytics.nn.modules.block.C2f',
    'ultralytics.nn.modules.conv.Conv',
    'ultralytics.nn.modules.block.SPPF',
    'ultralytics.nn.modules.block.Bottleneck'
]

logger = logging.getLogger("classroom_image_analyzer")
logging.basicConfig(level=logging.INFO)

class ClassroomImageAnalyzer:
    
    def __init__(self):
        self.setup_logging()
        self.load_models()
        self.subject_keywords = self.initialize_subject_mapping()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        try:
            import torch  
            # Force CPU for ViT and embedding models due to limited GPU memory
            self.device = torch.device('cpu')
            
           
            yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/yolo/yolo.pt'))
            print("Loading YOLO from:", yolo_path)
            
           
            for global_name in SAFE_GLOBALS:
                # Removed serialization.add_safe_globals([global_name]) as it is not a valid PyTorch method
                pass # This line was removed as per the edit hint.
            
            print(f"Added {', '.join(SAFE_GLOBALS)} to safe globals for YOLO loading")
            
            
            try:
                self.yolo_model = YOLO(yolo_path)
                print("YOLO model loaded successfully with default settings")
            except Exception as e:
                print(f"YOLO loading failed with default settings: {e}")
                print("Trying alternative loading method...")
                
                # Alternative: Load with weights_only=False
                try:
                    # Load directly with torch.load and weights_only=False for trusted source
                    checkpoint = torch.load(yolo_path, map_location='cpu')

                    print("YOLO checkpoint loaded successfully with weights_only=False")
                    
                    # Create YOLO model from checkpoint
                    self.yolo_model = YOLO(yolo_path)
                    print("YOLO model loaded successfully with weights_only=False")
                    
                except Exception as e2:
                    print(f"Alternative YOLO loading also failed: {e2}")
                    print("YOLO model will not be available")
                    self.yolo_model = None
            
           
            self.ocr_processor, self.ocr_model = self._load_ocr_models()
            
            # Load BLIP models with OpenVINO optimization
            self.blip_processor, self.blip_model = self._load_blip_models()
            
            self.vqa_pipeline = pipeline("visual-question-answering", 
                                       model="dandelin/vilt-b32-finetuned-vqa")

            # Load ViT model with improved class names handling
            vit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/vit/vit_finetuned.pth'))
            print("Loading ViT for classification and embeddings from:", vit_path)
            
            if os.path.exists(vit_path):
                try:
                    # Load class names from dataset directories first
                    landmark_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../dataset/landmark'))
                    history_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../dataset/History'))
                    class_names = set()
                    for base_dir in [landmark_dir, history_dir]:
                        if os.path.isdir(base_dir):
                            for entry in os.listdir(base_dir):
                                entry_path = os.path.join(base_dir, entry)
                                if os.path.isdir(entry_path):
                                    class_names.add(entry)
                    self.vit_classes = sorted(list(class_names))
                    print(f"Found {len(self.vit_classes)} class names from dataset: {self.vit_classes}")
                    # Load checkpoint
                    state_dict = torch.load(vit_path, map_location=self.device, weights_only=True)
                    # Always use dataset class names and count
                    num_classes = len(self.vit_classes)
                    print(f"Forcing ViT model to use {num_classes} classes from dataset.")
                    self.vit_model = vit_b_16(weights=None)
                    self.vit_model.heads.head = torch.nn.Linear(self.vit_model.heads.head.in_features, num_classes)
                    # Remove checkpoint head if shape mismatch
                    if 'heads.head.weight' in state_dict and state_dict['heads.head.weight'].shape[0] != num_classes:
                        print(f"Removing checkpoint head: {state_dict['heads.head.weight'].shape[0]} -> {num_classes} classes")
                        state_dict.pop('heads.head.weight', None)
                        state_dict.pop('heads.head.bias', None)
                    self.vit_model.load_state_dict(state_dict, strict=False)
                    self.vit_model.to(self.device)
                    self.vit_model.eval()
                    # Embedding model
                    self.embedding_model = vit_b_16(weights=None)
                    self.embedding_model.heads.head = torch.nn.Linear(self.embedding_model.heads.head.in_features, num_classes)
                    self.embedding_model.load_state_dict(state_dict, strict=False)
                    self.embedding_model.heads.head = torch.nn.Identity()
                    self.embedding_model.to(self.device)
                    self.embedding_model.eval()
                    print(f"ViT model loaded successfully with {num_classes} classes: {self.vit_classes}")
                except Exception as e:
                    print(f"Error loading ViT weights: {e}")
                    print("Falling back to pretrained ViT model")
                    self._load_pretrained_vit()
            else:
                print(f"ViT weights not found at {vit_path}")
                print("Falling back to pretrained ViT model")
                self._load_pretrained_vit()

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
    
    def _load_pretrained_vit(self):
        """Load pretrained ViT model as fallback"""
        try:
            print("Loading pretrained ViT model...")
            self.vit_model = vit_b_16(weights='IMAGENET1K_V1')
            self.vit_model.heads.head = torch.nn.Linear(self.vit_model.heads.head.in_features, 2)  # Default to 2 classes
            self.vit_model.to(self.device)
            self.vit_model.eval()
            
            self.embedding_model = vit_b_16(weights='IMAGENET1K_V1')
            self.embedding_model.heads.head = torch.nn.Identity()
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
            
            self.vit_classes = ['landmark', 'historical']  # Default classes
            print("Pretrained ViT model loaded successfully")
            
        except Exception as e:
            print(f"Error loading pretrained ViT: {e}")
            # Create a minimal ViT model as last resort
            self.vit_model = vit_b_16(weights=None)
            self.vit_model.heads.head = torch.nn.Linear(self.vit_model.heads.head.in_features, 2)
            self.vit_model.to(self.device)
            self.vit_model.eval()
            
            self.embedding_model = vit_b_16(weights=None)
            self.embedding_model.heads.head = torch.nn.Identity()
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
            
            self.vit_classes = ['landmark', 'historical']
            print("Minimal ViT model created as fallback")
            
    def initialize_subject_mapping(self) -> Dict[str, List[str]]:
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
                out = self.blip_model.generate(**inputs, max_length=config.MAX_RESPONSE_TOKENS, num_beams=3)
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
        text_content = image_analysis.get('text', '')
        objects = [d['class'] for d in image_analysis.get('objects', [])]
        caption = image_analysis.get('caption', '')
        subject = image_analysis.get('subject', 'general')
        
        if subject == 'history':
            return self.generate_historical_explanation(text_content, objects, caption)
        return self.generate_general_explanation(text_content, objects, caption)

    def generate_general_explanation(self, text: str, objects: List[str], caption: str) -> str:
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

    def latex_ocr_equation(self, image_path):
        if not latex_ocr_available:
            return None
        try:
            ocr = LatexOCR()
            result = ocr(image_path)
            return result
        except Exception as e:
            self.logger.error(f"[LaTeX-OCR] Error: {e}")
            return None

    def tesseract_ocr(self, image_path):
        try:
            img = PILImage.open(image_path)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            self.logger.error(f"[Tesseract-OCR] Error: {e}")
            return None

    def analyze_image_comprehensive(self, image_path: str, specific_questions: List[str] = None, user_text: str = None) -> Dict[str, Any]:
        self.logger.info(f"[ANALYZE] Starting comprehensive analysis for: {image_path}")
        self.logger.info(f"[ANALYZE] LaTeX-OCR available: {latex_ocr_available}")
        image = self.preprocess_image(image_path)
        if image is None:
            self.logger.error("[ANALYZE] Image preprocessing failed")
            return {'error': 'Could not process image'}
        try:
            # Try LaTeX-OCR for math, Tesseract for documents
            latex_ocr_text = self.latex_ocr_equation(image_path) if latex_ocr_available else None
            self.logger.info(f"[ANALYZE] LaTeX-OCR raw output: {latex_ocr_text}")
            if latex_ocr_text and (re.search(r'[=+\-*/^]', latex_ocr_text) or len(latex_ocr_text) > 3):
                ocr_text = latex_ocr_text
                self.logger.info(f"[ANALYZE] LaTeX-OCR text used: {ocr_text}")
            else:
                tesseract_text = self.tesseract_ocr(image_path)
                self.logger.info(f"[ANALYZE] Tesseract OCR output: {tesseract_text[:100] if tesseract_text else tesseract_text}")
                if tesseract_text and len(tesseract_text.split()) > 5:
                    ocr_text = tesseract_text
                    self.logger.info(f"[ANALYZE] Tesseract OCR text used: {ocr_text[:100]}")
                else:
                    ocr_text = self.extract_text_ocr(image)
                    self.logger.info(f"[ANALYZE] TrOCR text used: {ocr_text[:100]}")
            text_content = user_text if user_text else ocr_text

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
                # Fallback: If OCR text is present and looks like math/text, prioritize it
                if ocr_text and (re.search(r'[=+\-*/^]', ocr_text) or len(ocr_text) > 3):
                    final_prediction = {
                        'type': 'ocr_text',
                        'label': ocr_text,
                        'confidence': 0.7
                    }
                    self.logger.info(f"[ANALYZE] Final prediction: OCR (text/equation)")
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
            # If OCR text is prioritized, generate a math/text explanation
            if final_prediction['type'] == 'ocr_text':
                analysis_result['explanation'] = f"The image contains the following text or equation: {ocr_text}\n"
                # Optionally, add a simple math solver for equations
                if re.search(r'=', ocr_text):
                    try:
                        from sympy import Eq, symbols, solve
                        x = symbols('x')
                        eq = ocr_text.replace('^', '**')
                        eq = eq.replace('= 0', '') if '= 0' in eq else eq
                        eq = eq.replace('=0', '') if '=0' in eq else eq
                        eq = eq.replace(' ', '')
                        # Only handle simple quadratics for now
                        sol = solve(eq, x)
                        analysis_result['explanation'] += f"\nSolving for x: {sol}"
                    except Exception as e:
                        analysis_result['explanation'] += f"\n(Could not solve equation automatically: {e})"
            else:
                analysis_result['explanation'] = self.generate_educational_explanation(analysis_result)
            self.logger.info(f"[ANALYZE] Comprehensive analysis complete for: {image_path}")
            return analysis_result
        except Exception as e:
            self.logger.error(f"[ANALYZE] Error in comprehensive analysis: {e}")
            return {'error': str(e)}

    def generate_unified_explanation(self, classification: Dict[str, str], analysis: Dict[str, Any]) -> str:
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

    def _load_ocr_models(self):
        try:
            if OpenVINOConfig.should_use_openvino():
                try:
                    from optimum.intel import OVModelForSeq2SeqLM
                    openvino_cache_path = OpenVINOConfig.get_model_cache_path("trocr-base-printed")
                    if os.path.exists(openvino_cache_path) and not OpenVINOConfig.should_export_models():
                        logger.info(f"Loading OpenVINO optimized TrOCR from cache: {openvino_cache_path}")
                        model = OVModelForSeq2SeqLM.from_pretrained(openvino_cache_path)
                        processor = TrOCRProcessor.from_pretrained(openvino_cache_path)
                    else:
                        logger.info("Exporting TrOCR to OpenVINO format...")
                        os.makedirs(openvino_cache_path, exist_ok=True)
                        model = OVModelForSeq2SeqLM.from_pretrained(
                            OpenVINOConfig.TROCR_MODEL_NAME,
                            export=True
                        )
                        processor = TrOCRProcessor.from_pretrained(OpenVINOConfig.TROCR_MODEL_NAME)
                        model.save_pretrained(openvino_cache_path)
                        processor.save_pretrained(openvino_cache_path)
                        logger.info(f"OpenVINO TrOCR model saved to: {openvino_cache_path}")
                    logger.info("OpenVINO optimized TrOCR model loaded successfully")
                    return processor, model
                except Exception as e:
                    logger.error(f"OpenVINO TrOCR not supported or failed: {e}\nFalling back to PyTorch TrOCR model.")
                    return self._load_pytorch_ocr_models()
            else:
                return self._load_pytorch_ocr_models()
        except Exception as e:
            logger.error(f"Error loading OCR models: {e}")
            return self._load_pytorch_ocr_models()

    def _load_pytorch_ocr_models(self):
        """Load PyTorch TrOCR models (fallback)"""
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed', use_fast=True)
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        return processor, model
    
    def _load_blip_models(self):
        try:
            if OpenVINOConfig.should_use_openvino():
                try:
                    from optimum.intel import OVModelForSeq2SeqLM
                    openvino_cache_path = OpenVINOConfig.get_model_cache_path("blip-image-captioning-base")
                    if os.path.exists(openvino_cache_path) and not OpenVINOConfig.should_export_models():
                        logger.info(f"Loading OpenVINO optimized BLIP from cache: {openvino_cache_path}")
                        model = OVModelForSeq2SeqLM.from_pretrained(openvino_cache_path)
                        processor = BlipProcessor.from_pretrained(openvino_cache_path)
                    else:
                        logger.info("Exporting BLIP to OpenVINO format...")
                        os.makedirs(openvino_cache_path, exist_ok=True)
                        model = OVModelForSeq2SeqLM.from_pretrained(
                            OpenVINOConfig.BLIP_MODEL_NAME,
                            export=True
                        )
                        processor = BlipProcessor.from_pretrained(OpenVINOConfig.BLIP_MODEL_NAME)
                        model.save_pretrained(openvino_cache_path)
                        processor.save_pretrained(openvino_cache_path)
                        logger.info(f"OpenVINO BLIP model saved to: {openvino_cache_path}")
                    logger.info("OpenVINO optimized BLIP model loaded successfully")
                    return processor, model
                except Exception as e:
                    logger.error(f"OpenVINO BLIP not supported or failed: {e}\nFalling back to PyTorch BLIP model.")
                    return self._load_pytorch_blip_models()
            else:
                return self._load_pytorch_blip_models()
        except Exception as e:
            logger.error(f"Error loading BLIP models: {e}")
            return self._load_pytorch_blip_models()

    def _load_pytorch_blip_models(self):
        """Load PyTorch BLIP models (fallback)"""
        processor = BlipProcessor.from_pretrained(BLIP_LOCAL_PATH, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_LOCAL_PATH)
        return processor, model 