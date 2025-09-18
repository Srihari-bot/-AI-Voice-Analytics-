# 🎤 AI Voice Analytics PoC

> **Intelligent Voice-to-Insight Platform for Business Analytics**

A cutting-edge proof-of-concept application that transforms audio conversations into actionable business insights using advanced AI technologies including speech-to-text, natural language processing, and retrieval-augmented generation (RAG).

![AI Voice Analytics](https://img.shields.io/badge/AI-Voice%20Analytics-blue?style=for-the-badge&logo=artificial-intelligence)
![React](https://img.shields.io/badge/React-19.1.1-61dafb?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python)

## 🚀 What This Application Does

Transform your audio conversations into structured business insights with our AI-powered analytics platform. Perfect for:

- **Customer Support Analysis** - Analyze support calls to identify common issues and solutions
- **Business Meeting Intelligence** - Extract key decisions and action items from meetings
- **Training & Quality Assurance** - Review and improve communication patterns
- **Compliance Monitoring** - Track and analyze business conversations for regulatory compliance

## ✨ Key Features

### 🎯 **Multi-Model Speech Recognition**
- **Whisper AI** - OpenAI's state-of-the-art speech recognition
- **Kibo AI** - Advanced Indian language support
- **Assisto AI** - Specialized business conversation understanding
- **Real-time model switching** without restarting the application

### 🧠 **Intelligent Analysis Pipeline**
1. **Audio Transcription** - Convert speech to text with high accuracy
2. **Intent Recognition** - Understand the purpose and context of conversations
3. **Keyword & Entity Extraction** - Identify important business terms and concepts
4. **RAG-Enhanced Resolution** - Generate actionable insights using knowledge base

### 📊 **Dual View Interface**
- **Business View** - Clean, executive-friendly insights and recommendations
- **Diagnostic View** - Detailed technical analysis with confidence scores

### 🔗 **Smart Knowledge Integration**
- **Automatic hyperlink detection** - Convert URLs to clickable links
- **Markdown rendering** - Rich text formatting with proper styling
- **Knowledge base suggestions** - Relevant articles and documentation

### 🎨 **Modern User Experience**
- **Drag & drop file upload** - Intuitive audio file management
- **Real-time progress tracking** - Visual feedback during processing
- **Responsive design** - Works seamlessly on desktop and mobile
- **Error handling** - Graceful degradation and user-friendly error messages

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Services   │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Multiple)    │
│                 │    │                 │    │                 │
│ • File Upload   │    │ • Audio Proc.   │    │ • Whisper       │
│ • Real-time UI  │    │ • STT Models    │    │ • Kibo AI       │
│ • Markdown      │    │ • RAG Pipeline  │    │ • Assisto       │
│ • Analytics     │    │ • ChromaDB      │    │ • WatsonX       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Frontend
- **React 19.1.1** - Modern UI framework
- **Vite** - Lightning-fast build tool
- **Axios** - HTTP client for API communication
- **React Markdown** - Rich text rendering
- **CSS3** - Modern styling with custom components

### Backend
- **FastAPI 0.104.1** - High-performance Python web framework
- **Uvicorn** - ASGI server for production deployment
- **ChromaDB** - Vector database for RAG implementation
- **Pydantic** - Data validation and serialization

### AI & ML
- **OpenAI Whisper** - Speech-to-text transcription
- **IBM WatsonX** - Large language model for analysis
- **Transformers** - Hugging Face model integration
- **Librosa** - Audio processing and analysis

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Node.js 18+**
- **IBM WatsonX API Key** (for full functionality)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-voice-analytics-poc.git
cd ai-voice-analytics-poc
```

### 2. Backend Setup
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp env_template.txt .env

# Edit .env with your credentials
# WATSONX_API_KEY=your_api_key_here
# WATSONX_PROJECT_ID=your_project_id_here

# Start the backend server
python main.py
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

### 4. Access the Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📋 Usage Guide

### 1. Upload Audio Files
- **Supported formats**: MP3, WAV
- **Drag & drop** or click to select files
- **Multiple files** supported for batch processing

### 2. Select STT Model
- Click the **top-right corner** to access model selection
- Choose between **Whisper**, **Kibo AI**, or **Assisto**
- Switch models **without restarting** the application

### 3. Analyze Audio
- Select an audio file from the **Backend Collections**
- The system automatically processes through the pipeline:
  - 🎤 **Transcription** - Convert speech to text
  - 🎯 **Intent Recognition** - Understand conversation purpose
  - 🔍 **Keyword Extraction** - Identify important terms
  - 💡 **Resolution Generation** - Create actionable insights

### 4. View Results
- **Business View** - Executive summary with recommendations
- **Diagnostic View** - Technical details with confidence scores
- **Clickable links** - Access relevant knowledge base articles

## 🔧 Configuration

### Environment Variables
```bash
# IBM WatsonX Configuration
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# File Upload Limits
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=.mp3,.wav
```

### Model Configuration
- **Whisper**: Best for general speech recognition
- **Kibo AI**: Optimized for Indian languages and accents
- **Assisto**: Specialized for business conversations

## 📊 API Endpoints

### Core Analysis
- `POST /analyze` - Complete audio analysis pipeline
- `POST /transcribe` - Speech-to-text conversion
- `POST /recognize-intent` - Intent recognition
- `POST /extract-keywords` - Keyword and entity extraction
- `POST /generate-resolution` - RAG-enhanced insights

### Model Management
- `GET /stt-model` - Get current STT model
- `POST /stt-model` - Switch STT model
- `GET /test-rag-status` - Test RAG pipeline status

### File Management
- `POST /api/upload/audio` - Upload audio files
- `GET /collection/all` - List all audio collections
- `GET /collection/audio/{id}` - Get specific audio data

## 🎯 Use Cases

### Customer Support
- **Issue Analysis** - Identify common customer problems
- **Solution Tracking** - Monitor resolution effectiveness
- **Quality Assurance** - Review support agent performance

### Business Intelligence
- **Meeting Insights** - Extract key decisions and action items
- **Trend Analysis** - Identify recurring themes and patterns
- **Compliance Monitoring** - Track regulatory compliance

### Training & Development
- **Communication Analysis** - Improve conversation skills
- **Knowledge Gaps** - Identify areas needing training
- **Best Practices** - Learn from successful interactions

## 🔒 Security & Privacy

- **Local Processing** - Audio files processed locally when possible
- **Secure API Keys** - Environment variable protection
- **Data Encryption** - Secure transmission and storage
- **Privacy Compliance** - GDPR and data protection ready

## 🚀 Deployment

### Development
```bash
# Backend
cd backend && python main.py

# Frontend
cd frontend && npm run dev
```

### Production
```bash
# Build frontend
cd frontend && npm run build

# Serve with production server
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker (Optional)
```bash
# Build and run with Docker
docker-compose up --build
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the Whisper speech recognition model
- **IBM** for WatsonX AI services
- **Hugging Face** for transformer models
- **FastAPI** team for the excellent web framework
- **React** team for the modern UI framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-voice-analytics-poc/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-voice-analytics-poc/discussions)
- **Email**: support@yourcompany.com

## 🔮 Roadmap

- [ ] **Real-time streaming** audio processing
- [ ] **Multi-language support** for global deployment
- [ ] **Advanced analytics** dashboard
- [ ] **Mobile app** for on-the-go analysis
- [ ] **Integration APIs** for third-party systems
- [ ] **Custom model training** for domain-specific use cases

---

<div align="center">

**Built with ❤️ for the future of voice analytics**

[⭐ Star this repo](https://github.com/yourusername/ai-voice-analytics-poc) | [🐛 Report Bug](https://github.com/yourusername/ai-voice-analytics-poc/issues) | [💡 Request Feature](https://github.com/yourusername/ai-voice-analytics-poc/issues)

</div>
