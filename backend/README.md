# Classroom Image Analysis System

A comprehensive image analysis system for educational content, supporting multiple subjects and providing detailed explanations.

## Features

- Object detection using YOLOv8
- Text extraction using TrOCR
- Image captioning using BLIP
- Visual question answering using ViLT
- Subject-specific analysis for:
  - Mathematics
  - Science
  - Geography
  - History
  - Language
- Interactive question generation
- Hardware optimization recommendations

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For CUDA support (if you have an NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Hardware Requirements

### Minimum Requirements
- 8GB RAM
- CPU processing (slower)
- 10GB free disk space

### Recommended Requirements
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 20GB free disk space

### Optimal Requirements
- 32GB RAM
- NVIDIA RTX 3080+ or Intel Arc GPU
- 30GB free disk space

## Usage

1. Start the server:
```bash
uvicorn app.main:app --reload
```

2. The server will be available at `http://localhost:8000`

3. API Endpoints:
   - POST `/api/image/analyze`: Upload and analyze images
   - GET `/api/image/status`: Check system status and hardware info

## Model Information

The system uses the following models:
- YOLOv8n for object detection
- TrOCR for text extraction
- BLIP for image captioning
- ViLT for visual question answering

Models will be downloaded automatically on first use.

## Optimization Tips

1. For Intel Hardware:
   - Install OpenVINO for acceleration
   - Use model quantization

2. For Real-time Processing:
   - Use quantized models
   - Enable half-precision inference

3. For Memory Constraints:
   - Use smaller models
   - Enable lazy loading
   - Adjust batch size

4. For Better Accuracy:
   - Use larger YOLO models (yolov8m.pt, yolov8l.pt)
   - Increase confidence thresholds

## Troubleshooting

1. If models fail to download:
   - Check internet connection
   - Verify disk space
   - Check model cache directory permissions

2. If CUDA errors occur:
   - Verify CUDA installation
   - Check GPU compatibility
   - Try CPU-only mode

3. If memory errors occur:
   - Reduce batch size
   - Use smaller models
   - Enable model quantization

## Contributing

Feel free to submit issues and enhancement requests! 