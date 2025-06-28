from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import torch
import io
import os
import uuid
from app.utils.vit_model import load_vit_model, predict_vit_class
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


try:
    
    yolo_model = YOLO('backend/models/yolo/best.pt')
    print("YOLO model loaded successfully")
    
    
    vit_model, vit_classes = load_vit_model("models/vit/vit_landmark_history.pth")
    print("ViT model loaded successfully")
    
except Exception as e:
    print(f"Error loading models: {e}")
    yolo_model = None
    vit_model = None
    vit_classes = []

@router.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify image using both YOLO and ViT models
    Returns the best prediction based on confidence scores
    """
    try:
        logger.info(f"/classify-image endpoint called. Filename: {file.filename}")
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        logger.info("Image loaded for classification.")
        
        results = {
            "yolo_prediction": None,
            "vit_prediction": None,
            "final_prediction": None,
            "confidence": 0.0,
            "model_used": None
        }
        
        # Run YOLO detection (for lab apparatus)
        if yolo_model:
            try:
                yolo_results = yolo_model.predict(img, conf=0.3)  # Lower confidence threshold
                
                if yolo_results and len(yolo_results) > 0:
                    result = yolo_results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Get the detection with highest confidence
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        max_conf_idx = confidences.argmax()
                        max_confidence = confidences[max_conf_idx]
                        detected_class = result.names[int(classes[max_conf_idx])]
                        
                        results["yolo_prediction"] = {
                            "class": detected_class,
                            "confidence": float(max_confidence),
                            "detections": len(result.boxes)
                        }
                        
                        logger.info(f"YOLO detected: {detected_class} with confidence {max_confidence}")
            except Exception as e:
                logger.error(f"YOLO prediction error: {e}")
        
        # Run ViT classification (for landmarks/historical)
        if vit_model and vit_classes:
            try:
                vit_result = predict_vit_class(img, vit_model, vit_classes)
                
                if vit_result:
                    results["vit_prediction"] = vit_result
                    logger.info(f"ViT predicted: {vit_result['class']} with confidence {vit_result['confidence']}")
            except Exception as e:
                logger.error(f"ViT prediction error: {e}")
        
        # Determine final prediction based on confidence
        yolo_conf = results["yolo_prediction"]["confidence"] if results["yolo_prediction"] else 0.0
        vit_conf = results["vit_prediction"]["confidence"] if results["vit_prediction"] else 0.0
        
        if yolo_conf > vit_conf and yolo_conf > 0.5:
            # Use YOLO prediction if it's more confident
            results["final_prediction"] = results["yolo_prediction"]["class"]
            results["confidence"] = yolo_conf
            results["model_used"] = "YOLO"
            results["category"] = "laboratory_apparatus"
        elif vit_conf > 0.3:
            # Use ViT prediction if it's confident enough
            results["final_prediction"] = results["vit_prediction"]["class"]
            results["confidence"] = vit_conf
            results["model_used"] = "ViT"
            results["category"] = "landmark_historical"
        else:
           
            results["final_prediction"] = "unknown"
            results["confidence"] = max(yolo_conf, vit_conf)
            results["model_used"] = "none"
            results["category"] = "unknown"
        
        logger.info(f"Classification result: {results}")
        return JSONResponse(results)
        
    except Exception as e:
        logger.error(f"Error in /classify-image endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/classify-image-detailed")
async def classify_image_detailed(file: UploadFile = File(...)):
    """
    Detailed classification with all model outputs
    """
    try:
        logger.info(f"/classify-image-detailed endpoint called. Filename: {file.filename}")
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        logger.info("Image loaded for detailed classification.")
        
        results = {
            "yolo_detections": [],
            "vit_classification": None,
            "summary": {}
        }
        
        
        if yolo_model:
            try:
                yolo_results = yolo_model.predict(img, conf=0.3)
                
                if yolo_results and len(yolo_results) > 0:
                    result = yolo_results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        for i in range(len(result.boxes)):
                            detection = {
                                "class": result.names[int(result.boxes.cls[i])],
                                "confidence": float(result.boxes.conf[i]),
                                "bbox": result.boxes.xyxy[i].cpu().numpy().tolist()
                            }
                            results["yolo_detections"].append(detection)
                        logger.info(f"YOLO detailed detections: {results['yolo_detections']}")
            except Exception as e:
                logger.error(f"YOLO detailed prediction error: {e}")
        
        
        if vit_model and vit_classes:
            try:
                vit_result = predict_vit_class(img, vit_model, vit_classes)
                results["vit_classification"] = vit_result
                logger.info(f"ViT detailed classification: {vit_result}")
            except Exception as e:
                logger.error(f"ViT detailed prediction error: {e}")
        
        
        if results["yolo_detections"]:
            results["summary"]["has_lab_apparatus"] = True
            results["summary"]["lab_apparatus_count"] = len(results["yolo_detections"])
            results["summary"]["highest_confidence_detection"] = max(
                results["yolo_detections"], 
                key=lambda x: x["confidence"]
            )
        else:
            results["summary"]["has_lab_apparatus"] = False
        
        if results["vit_classification"]:
            results["summary"]["is_landmark_or_historical"] = True
            results["summary"]["landmark_type"] = results["vit_classification"]["class"]
            results["summary"]["landmark_confidence"] = results["vit_classification"]["confidence"]
        else:
            results["summary"]["is_landmark_or_historical"] = False
        
        logger.info(f"Detailed classification summary: {results['summary']}")
        return JSONResponse(results)
        
    except Exception as e:
        logger.error(f"Error in /classify-image-detailed endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}") 