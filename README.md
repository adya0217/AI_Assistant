# Multimodal Classroom Assistant

A comprehensive multimodal AI assistant for educational content, supporting image, audio, and text analysis with detailed explanations. Combines advanced chaining techniques (LangChain) for educational support through text, voice, and image analysis.

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (recommended: 3.9 or 3.10)
- **Node.js 16+** and npm
- **Git**
- **8GB+ RAM** (16GB+ recommended)
- **10GB+ free disk space** (20GB+ recommended)

### System Requirements

#### Minimum Requirements
- 8GB RAM
- CPU processing (slower)
- 10GB free disk space
- Windows 10/11, macOS 10.15+, or Ubuntu 18.04+

#### Recommended Requirements
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM (optional but recommended)
- 20GB free disk space
- SSD storage

#### Optimal Requirements
- 32GB RAM
- NVIDIA RTX 3080+ or Intel Arc GPU
- 30GB free disk space
- NVMe SSD

## 🛠️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd AI_Assistant
```

### Step 2: Backend Setup

#### 2.1 Create Python Virtual Environment
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2.2 Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
# (Optional) For CUDA support (NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Install additional dependencies for LangChain
pip install langchain langchain-ollama langchain-core
# Install FastAPI and Uvicorn
pip install fastapi uvicorn python-multipart
```

#### 2.3 Download AI Models
```bash
python download_models.py
```

#### 2.4 Install System Dependencies

**On Windows:**
- Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR`
- (If needed) Install Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**On macOS:**
```bash
brew install tesseract
brew install libmagic
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
sudo apt-get install -y libmagic1
sudo apt-get install -y ffmpeg
```

### Step 3: Frontend Setup

#### 3.1 Install Node.js Dependencies
```bash
cd ../frontend
npm install
```

#### 3.2 Configure Frontend
The frontend is configured to proxy requests to `http://localhost:8000` (backend). This is already set in `package.json`.

### Step 4: Environment Configuration

#### 4.1 Create Environment File
```bash
cd ../backend
# Create .env file (if not exists)
cp .env.example .env  # if .env.example exists
```

#### 4.2 Configure Environment Variables
Create or edit `.env` file in the backend directory:

```env
# Required (if using Supabase)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key

# Optional (with defaults)
EMBEDDING_RETENTION_DAYS=30
DOCUMENT_RETENTION_DAYS=90
CONTEXT_HISTORY_LIMIT=50
MAX_QUERY_LENGTH=1000
MAX_FILE_SIZE_MB=10
MAX_IMAGE_DIMENSION=2048
LLM_MODEL=mistral
STT_MODEL=base
EMBEDDING_MODEL=all-MiniLM-L6-v2
BACKGROUND_TASK_TIMEOUT=300
CLEANUP_INTERVAL_HOURS=24
```

## Database (Supabase) Setup

This project uses [Supabase](https://supabase.com/) as a vector database for RAG and caching.

**Required:**
- A Supabase project (free tier is fine)
- The following tables:
  - `user_queries` (for storing user queries and embeddings)
  - `documents` (for storing document chunks and embeddings)
  - `image_embeddings` (for storing image embeddings)
- The following RPC functions:
  - `match_queries` (for vector similarity search on queries)
  - `match_images` (for vector similarity search on images)

**Table schemas and RPCs:**
- See the code in `backend/app/api/fusion.py`, `backend/app/api/pdf_routes.py`, and `backend/app/services/llm_chain.py` for expected fields.

### Example: user_queries Table (with pgvector)

> **Note:** The `embedding` column uses the [pgvector](https://github.com/pgvector/pgvector) extension. You must enable this extension in your Supabase/Postgres instance. The vector dimension should match your embedding model (e.g., 384 for MiniLM-L6-v2).

```sql
-- Enable pgvector extension (run once per database)
create extension if not exists vector;

create table public.user_queries (
  id uuid not null default gen_random_uuid (),
  query text null,
  embedding vector(384) null, -- adjust dimension if needed
  role text null,
  timestamp timestamp with time zone null default timezone ('utc'::text, now()),
  constraint user_queries_pkey primary key (id)
) TABLESPACE pg_default;

create index if not exists user_queries_embedding_idx
  on public.user_queries
  using ivfflat (embedding vector_cosine_ops) TABLESPACE pg_default;
```
- Adjust the vector dimension (e.g., `vector(384)`) to match your embedding model.
- The `ivfflat` index is required for fast vector similarity search.
- Make sure to enable the `pgvector` extension in your database.

- Example schema for `documents`:
  - `id` (uuid, primary key)
  - `content` (text)
  - `embedding` (vector/float8[] or text[])
  - `created_at` (timestamp)

**You must create these tables and functions in your Supabase project.**

## 🚀 Running the Application

### Step 1: Start the Backend Server
```bash
cd backend
# Activate virtual environment (if not already activated)
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The backend will be available at `http://localhost:8000`

### Step 2: Start the Frontend Development Server
```bash
cd frontend
npm start
```
The frontend will be available at `http://localhost:3000`

### Step 3: Access the Application
Open your browser and navigate to `http://localhost:3000`

## 🔧 API Endpoints

### Unified Multimodal Endpoint
```
POST /api/unified/analyze
```
**Handles**: Text, voice, and image inputs in a single request
**Parameters**:
- `text_query`: Text input
- `voice_file`: Audio file
- `image_file`: Image file
- `custom_prompt`: Optional custom prompt

### Individual Endpoints
```
POST /api/ask_text          # Text-only processing
POST /api/audio/transcribe_audio/ # Voice processing
POST /api/image/analyze     # Image processing
POST /api/pdf/upload_pdf/   # PDF upload and analysis
POST /api/chat              # General chat
POST /api/multimodal        # Combined multimodal
```

### Context Management
```
GET  /api/context/summary   # Get context summary
POST /api/context/clear     # Clear context
GET  /api/config            # Get system configuration
```

### Maintenance
```
POST /api/maintenance/cleanup  # Trigger cleanup of old embeddings
```

## 🎯 Key Features

- Object detection using YOLOv8
- Text extraction using TrOCR, LaTeX-OCR, Tesseract
- Image captioning using BLIP
- Visual question answering using ViLT
- Audio transcription using Whisper
- Retrieval-augmented generation (RAG) with Supabase
- Subject-specific analysis (Math, Science, etc.)
- Interactive question generation
- Hardware optimization recommendations
- Multimodal input: text, audio, image, PDF
- Contextual LLM prompting and chaining

## Model Information

The system uses the following models (auto-downloaded on first use):
- YOLOv8n for object detection
- TrOCR, LaTeX-OCR, Tesseract for OCR
- BLIP for image captioning
- ViLT for visual question answering
- Whisper for audio transcription
- all-MiniLM-L6-v2 for embeddings

## Optimization Tips

- For Intel Hardware: Install OpenVINO for acceleration
- For Real-time Processing: Use quantized models, enable half-precision inference
- For Memory Constraints: Use smaller models, enable lazy loading, adjust batch size
- For Better Accuracy: Use larger YOLO models, increase confidence thresholds

## Troubleshooting

- **Models fail to download:**
  - Check internet connection
  - Verify disk space
  - Check model cache directory permissions

- **CUDA errors:**
  - Verify CUDA installation
  - Check GPU compatibility
  - Try CPU-only mode

- **Memory errors:**
  - Reduce batch size
  - Use smaller models
  - Enable model quantization

- **Supabase/database errors:**
  - Ensure `.env` has correct `SUPABASE_URL` and `SUPABASE_KEY`
  - Make sure required tables and RPCs exist in your Supabase project
  - Check Supabase RLS (Row Level Security) policies allow inserts/selects for your service key
  - See logs for specific error messages

## Contributing

Feel free to submit issues and enhancement requests! 