import os
from pathlib import Path
from transformers import AutoModel, BlipForConditionalGeneration

MODELS_DIR = Path(__file__).parent / 'models'

# Download TrOCR
print('Downloading TrOCR...')
trocr_dir = MODELS_DIR / 'trocr-base-printed'
trocr_dir.mkdir(parents=True, exist_ok=True)
AutoModel.from_pretrained('microsoft/trocr-base-printed', cache_dir=str(trocr_dir))

# Download BLIP
print('Downloading BLIP...')
blip_dir = MODELS_DIR / 'blip-image-captioning-base'
blip_dir.mkdir(parents=True, exist_ok=True)
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir=str(blip_dir))

# Download all-MiniLM-L6-v2
print('Downloading all-MiniLM-L6-v2...')
minilm_dir = MODELS_DIR / 'all-MiniLM-L6-v2'
minilm_dir.mkdir(parents=True, exist_ok=True)
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir=str(minilm_dir))

print('All models downloaded.') 