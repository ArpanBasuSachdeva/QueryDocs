# QueryDocs - Intelligent Document Chat System

QueryDocs is an advanced document management and AI-powered chat system that allows users to upload documents, interact with them through an AI interface, and maintain detailed chat histories. The system uses document embeddings and intelligent retrieval to provide context-aware responses.

## 🌟 Features

### 1. Document Management
- Upload various document types (txt, md, csv, pdf)
- Configurable chunk size and overlap for document processing
- Document status tracking and management
- Hash-based document identification
- Detailed document metadata storage

### 2. AI-Powered Chat Interface
- Context-aware chat with specific documents
- Document-specific chat history
- Real-time AI responses
- Session-based chat management
- Support for multiple concurrent users

### 3. Chat History
- Comprehensive chat history tracking
- Session-based chat organization
- Document-specific chat filtering
- Timestamp and user tracking
- Detailed conversation logging

### 4. Exception Handling
- Detailed error logging
- Exception tracking and monitoring
- User-friendly error messages
- System health monitoring

## 🛠️ Technical Stack

### Backend
- FastAPI (Python web framework)
- SQLAlchemy (ORM)
- Uvicorn (ASGI server)
- Pydantic (Data validation)

### Frontend
- Streamlit (Web interface)
- Pandas (Data handling)
- Requests (API communication)

### Database
- SQLite (Default database)
- Support for PostgreSQL (Configurable)

### AI/ML Components
- Document embedding generation
- Semantic search capabilities
- Context-aware response generation

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/querydocs.git
cd querydocs
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the FastAPI backend:
```bash
uvicorn main:app --reload
```

2. In a separate terminal, start the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

3. Access the application:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📁 Project Structure

```
CopyHaiJi/
├── main.py                 # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Project dependencies
├── Router/
│   ├── table_creater.py   # Database models
│   ├── relations.py       # Database relations
│   ├── Chatbot_retriver.py # AI chat functionality
│   ├── exception_utils.py # Error handling
│   └── embedding.py       # Document embedding
├── uploads/               # Document storage
└── README.md             # Project documentation
```

## 🔄 API Endpoints

### Document Management
- `GET /documents` - List all documents
- `POST /upload-document` - Upload new document
- `GET /documents/{document_id}` - Get document details
- `DELETE /documents/{document_id}` - Delete document

### Chat
- `POST /chat` - Send chat message
- `GET /chat/history` - Get chat history
- `GET /chat/history/{user_id}` - Get user-specific chat history

### System
- `GET /health` - System health check
- `GET /api/exceptions/table` - View exception logs
- `DELETE /api/exceptions/cleanup` - Clean up old exceptions

## 🔒 Security Features

- Session-based user management
- Document access control
- Secure file handling
- Input validation
- Error logging and monitoring

## 🧪 Testing

Run tests using pytest:
```bash
pytest
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- Streamlit team for the amazing frontend framework
- All contributors and users of the project

## 🔄 Future Enhancements

- [ ] User authentication and authorization
- [ ] Real-time chat updates
- [ ] Document version control
- [ ] Advanced search capabilities
- [ ] Multi-language support
- [ ] API rate limiting
- [ ] Enhanced error handling
- [ ] Performance optimizations 