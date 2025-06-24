# Multimodal AI Assistant - Complete Chaining System

A comprehensive AI assistant that combines multiple input modalities using advanced chaining techniques with LangChain, providing educational support through text, voice, and image analysis.

## 🚀 System Architecture

### Multimodal Chaining Pipeline

```
Voice Input: Whisper → Text
Image Input: OpenCV/YOLO/VIT → Description/Text  
Text Input: Normal chat
Context Manager: ChatGPT-style memory
LangChain: Custom prompt control
React Frontend: UI for multimodal inputs
```

## 🏗️ Core Components

### 1. **Multimodal Chain (`backend/app/services/llm_chain.py`)**
- **Purpose**: Central orchestrator for all input types
- **Features**:
  - Voice processing with Whisper transcription
  - Image analysis with OpenCV/YOLO/ViT
  - Text processing with context awareness
  - Educational prompt templates
  - Conversation history management

### 2. **Context Manager (`backend/app/services/context_manager.py`)**
- **Purpose**: ChatGPT-style memory system
- **Features**:
  - Conversation history tracking (configurable limit)
  - Image context storage with retention policies
  - Voice session management
  - Context summarization and statistics
  - Export/import capabilities
  - Automatic cleanup of old entries

### 3. **System Configuration (`backend/app/config/system_config.py`)**
- **Purpose**: Centralized configuration management
- **Features**:
  - Retention policy settings
  - Input validation limits
  - File type restrictions
  - Model configuration
  - Background task settings

### 4. **Image Analyzer (`backend/app/services/classroom_image_analyzer.py`)**
- **Purpose**: Comprehensive image understanding
- **Models Used**:
  - YOLO: Object detection
  - ViT: Landmark/historical classification
  - BLIP: Image captioning
  - TrOCR: Text extraction
  - VQA: Visual question answering

### 5. **Voice Processing (`backend/app/services/whisper_stt.py`)**
- **Purpose**: Speech-to-text conversion
- **Features**:
  - Real-time audio recording
  - Whisper model integration
  - Audio file transcription

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
POST /api/ask/voice         # Voice processing
POST /api/ask/image         # Image processing
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

### 1. **Multimodal Input Processing**
- **Text**: Direct query processing with context awareness
- **Voice**: Whisper STT → Text → LLM response
- **Image**: Computer vision analysis → LLM explanation
- **Combined**: Multiple input types in single request

### 2. **Educational Focus**
- Subject-specific keyword mapping
- Historical landmark recognition
- Laboratory apparatus detection
- Educational prompt templates
- Context-aware responses

### 3. **Advanced Context Management**
- Conversation history (configurable limit, default: 50)
- Image context storage with timestamps
- Voice session tracking
- Cross-modal context awareness
- Session persistence
- Automatic cleanup of old entries

### 4. **LangChain Integration**
- Custom prompt templates for each input type
- Ollama LLM integration (Mistral model)
- Chain-based processing
- Educational response generation

### 5. **Production-Ready Features**
- **Background Tasks**: Non-blocking operations for embeddings and cleanup
- **Input Sanitization**: Query normalization and length limits
- **File Validation**: Type and size restrictions
- **Retention Policies**: Automatic cleanup of old data
- **Environment Validation**: Runtime configuration checks
- **Consistent Response Models**: Pydantic models for all endpoints

## 🛠️ Installation & Setup

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Install LangChain dependencies
pip install langchain langchain-ollama langchain-core

# Set up environment variables
cp .env.example .env
# Edit .env with your Supabase credentials
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Model Setup
1. **YOLO Model**: Place `best.pt` in `backend/models/yolo/`
2. **ViT Model**: Place `vit_landmark_history.pth` in `backend/models/vit/`
3. **Whisper**: Automatically downloaded on first use

## ⚙️ Configuration

### Environment Variables
```env
# Required
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

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

### System Configuration
The system uses a centralized configuration system (`SystemConfig`) that manages:
- **Retention Policies**: Automatic cleanup of old embeddings and documents
- **Input Limits**: Query length, file size, and image dimension limits
- **File Type Validation**: Allowed file types for uploads
- **Model Settings**: Configurable model parameters
- **Background Task Settings**: Timeout and cleanup intervals

## 📊 Database Schema

### Supabase Tables

#### `documents` (for PDF processing)
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### `user_queries` (for personalization)
```sql
CREATE TABLE user_queries (
    id SERIAL PRIMARY KEY,
    query TEXT,
    embedding vector(384),
    timestamp TIMESTAMP DEFAULT NOW()
);
```

## 🔄 Processing Flow

### 1. **Voice Input Flow**
```
Audio File → File Validation → Whisper STT → Text Sanitization → Context Manager → LangChain → Response
```

### 2. **Image Input Flow**
```
Image File → File Validation → OpenCV Preprocessing → YOLO Detection → ViT Classification → 
BLIP Captioning → TrOCR Text Extraction → Context Manager → LangChain → Response
```

### 3. **Text Input Flow**
```
Text Query → Input Sanitization → Background Task (Embedding Storage) → Context Manager → LangChain → Response
```

### 4. **Multimodal Flow**
```
Multiple Inputs → Parallel Processing → Input Validation → Context Fusion → LangChain → Response
```

## 🛡️ Security & Validation

### Input Sanitization
- Query length limits (configurable, default: 1000 chars)
- Whitespace normalization
- Newline replacement
- Multiple space removal

### File Validation
- **Image Types**: .jpg, .jpeg, .png, .webp, .gif
- **Audio Types**: .wav, .mp3, .m4a, .webm
- **Document Types**: .pdf, .txt, .doc, .docx
- File size limits (configurable)

### Environment Validation
- Runtime checks for required environment variables
- Graceful error handling for missing configuration

## 🔧 Background Tasks

### Automatic Cleanup
- **Embedding Retention**: Configurable retention period (default: 30 days)
- **Document Retention**: Separate retention for documents (default: 90 days)
- **Context Cleanup**: Automatic cleanup of old conversation history

### Non-blocking Operations
- **Embedding Storage**: Background task for storing query embeddings
- **Cleanup Tasks**: Scheduled cleanup of old data
- **Logging**: Non-blocking logging operations

## 🎨 Frontend Features

### Modern UI Components
- **Multimodal Input Area**: Text, voice, and image upload
- **Real-time Processing**: Loading states and progress indicators
- **Context Display**: Analysis summaries and input type indicators
- **Responsive Design**: Mobile-friendly interface
- **Educational Theme**: Learning-focused styling

### Input Type Indicators
- 🎤 Voice input with transcription display
- 📷 Image input with object detection summary
- 📝 Text input with context awareness

## 🚀 Usage Examples

### 1. **Text Query**
```javascript
// Frontend
const response = await axios.post('/api/ask_text', {
  query: "What is photosynthesis?"
});
```

### 2. **Voice Input**
```javascript
// Frontend
const formData = new FormData();
formData.append('voice_file', audioBlob);
const response = await axios.post('/api/ask/voice', formData);
```

### 3. **Image Analysis**
```javascript
// Frontend
const formData = new FormData();
formData.append('image_file', imageFile);
formData.append('text_query', 'What do you see in this image?');
const response = await axios.post('/api/ask/image', formData);
```

### 4. **Multimodal Input**
```javascript
// Frontend
const formData = new FormData();
formData.append('text_query', 'Explain this image');
formData.append('image_file', imageFile);
formData.append('voice_file', audioBlob);
const response = await axios.post('/api/unified/analyze', formData);
```

## 🔍 Debugging & Monitoring

### Context Summary
```bash
curl http://localhost:8000/api/context/summary
```

### System Configuration
```bash
curl http://localhost:8000/api/config
```

### Capabilities Check
```bash
curl http://localhost:8000/api/unified/capabilities
```

### Manual Cleanup
```bash
curl -X POST http://localhost:8000/api/maintenance/cleanup
```

### Logs
- Backend logs in console with detailed processing information
- Frontend console logs for API calls and responses

## 🎯 Educational Applications

### 1. **History Education**
- Landmark recognition and historical context
- Artifact analysis and cultural significance
- Timeline-based learning

### 2. **Science Education**
- Laboratory apparatus identification
- Experiment documentation
- Scientific concept explanation

### 3. **Language Learning**
- Text extraction and translation
- Pronunciation feedback
- Contextual vocabulary building

### 4. **General Learning**
- Multimodal question answering
- Interactive explanations
- Personalized learning paths

## 🔮 Future Enhancements

### Planned Features
- **Streaming Responses**: Real-time response generation
- **Video Processing**: Video analysis capabilities
- **Multi-language Support**: Internationalization
- **Advanced Analytics**: Learning progress tracking
- **Integration APIs**: LMS and educational platform integration

### Technical Improvements
- **Model Optimization**: Faster inference times
- **Caching System**: Response caching for common queries
- **Scalability**: Horizontal scaling support
- **Security**: Enhanced authentication and authorization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation
- Review the code comments
- Open an issue on GitHub
- Contact the development team

---

**Built with ❤️ for educational technology advancement** 