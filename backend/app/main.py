import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import transformers
transformers.utils.logging.set_verbosity_error()

import torch
from ultralytics.nn.modules import Conv
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
import ultralytics.nn.tasks
import torch.nn.modules.container


torch.serialization.add_safe_globals([
    Conv,
    Conv2d,
    BatchNorm2d,
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential
])

import logging
import huggingface_hub

# Increase HuggingFace download timeout (default 10s too low for slow networks)
huggingface_hub.constants.HUGGINGFACE_HUB_DEFAULT_TIMEOUT = 120  # seconds

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import fusion, image_routes, unified_analysis, pdf_routes, audio_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal AI Assistant")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update as per your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Registering routers...")
app.include_router(fusion.router, prefix="/api")
logger.info("Fusion router registered at /api")
app.include_router(unified_analysis.router, prefix="/api/unified", tags=["unified"])
logger.info("Unified analysis router registered at /api/unified")
app.include_router(pdf_routes.router)
logger.info("PDF routes registered")
app.include_router(audio_routes.router)
logger.info("Audio routes registered")
app.include_router(image_routes.router, prefix="/api")
logger.info("Image routes registered at /api")

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Multimodal AI Assistant is live!"}
