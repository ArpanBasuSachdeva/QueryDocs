from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from Router.table_creater import engine, get_db, Base
from Router.relations import ExceptionLog, ChatbotLog, Embedding
import uvicorn
import os
from pathlib import Path
import shutil

# Import Router modules
from Router.Chatbot_retriver import augmented_retrieval
from Router.exception_utils import log_exception
from Router.embedding import DocumentLoaderManager
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime

class DocumentUploadResponse(BaseModel):
    message: str
    file_path: str
    filename: str
    chunk_size: int
    chunk_overlap: int
    embedding_id: int

# =============================================================================
# ROOT AND HEALTH ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {"message": "Welcome to CopyHaiJi Chat System 💀"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Chat system operational"
    }

# =============================================================================
# UNIFIED CHAT SYSTEM
# =============================================================================

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """
    Simple and clean chat interface for interacting with the LLM.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CopyHaiJi Chat</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .chat-container {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .user-controls {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                align-items: center;
            }
            .user-controls input {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 100px;
            }
            .user-controls button {
                padding: 8px 15px;
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .user-controls button:hover {
                background-color: #218838;
            }
            .chat-messages {
                height: 450px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                background-color: #fafafa;
            }
            .message {
                margin: 10px 0;
                padding: 12px;
                border-radius: 8px;
                max-width: 70%;
            }
            .user-message {
                background-color: #007bff;
                color: white;
                text-align: right;
                margin-left: auto;
                border-bottom-right-radius: 3px;
            }
            .bot-message {
                background-color: #e9ecef;
                color: #333;
                margin-right: auto;
                border-bottom-left-radius: 3px;
            }
            .message-time {
                font-size: 0.8em;
                opacity: 0.7;
                margin-top: 5px;
            }
            .input-container {
                display: flex;
                gap: 10px;
            }
            #messageInput {
                flex: 1;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            #sendButton {
                padding: 12px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            #sendButton:hover {
                background-color: #0056b3;
            }
            #sendButton:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            .loading {
                color: #666;
                font-style: italic;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .error {
                background-color: #f8d7da !important;
                color: #721c24 !important;
                border: 1px solid #f5c6cb;
            }
            .system-message {
                background-color: #d1ecf1 !important;
                color: #0c5460 !important;
                border: 1px solid #bee5eb;
                text-align: center;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>💀 CopyHaiJi Chat 💀</h1>
            
            <div class="user-controls">
                <label for="userIdInput">User ID:</label>
                <input type="number" id="userIdInput" value="1" min="1">
                <button onclick="loadChatHistory()">Load History</button>
                <button onclick="clearChat()">Clear Chat</button>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    <div>Hello! I'm your AI assistant. Ask me anything!</div>
                    <div class="message-time">Just now</div>
                </div>
            </div>
            
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
                <button id="sendButton" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            function getCurrentUserId() {
                return parseInt(document.getElementById('userIdInput').value) || 1;
            }
            
            function formatTime(timestamp) {
                const date = new Date(timestamp);
                return date.toLocaleTimeString();
            }
            
            function addMessage(content, isUser = false, timestamp = null, isError = false, isSystem = false) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                
                let className = 'message ';
                if (isError) {
                    className += 'bot-message error';
                } else if (isSystem) {
                    className += 'system-message';
                } else if (isUser) {
                    className += 'user-message';
                } else {
                    className += 'bot-message';
                }
                
                messageDiv.className = className;
                
                const timeStr = timestamp ? formatTime(timestamp) : formatTime(new Date());
                messageDiv.innerHTML = `
                    <div>${content}</div>
                    <div class="message-time">${timeStr}</div>
                `;
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                return messageDiv;
            }
            
            async function sendMessage() {
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                
                const message = messageInput.value.trim();
                if (!message) return;
                
                const userId = getCurrentUserId();
                
                // Add user message to chat
                addMessage(message, true);
                
                // Clear input and disable button
                messageInput.value = '';
                sendButton.disabled = true;
                sendButton.textContent = 'Thinking...';
                
                // Add loading message
                const loadingDiv = addMessage('AI is thinking...', false, null, false, false);
                loadingDiv.classList.add('loading');
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            user_id: userId
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Remove loading message
                    loadingDiv.remove();
                    
                    if (response.ok) {
                        // Add bot response
                        addMessage(data.response, false, data.timestamp);
                    } else {
                        // Add error message
                        addMessage('Error: ' + (data.detail || 'Something went wrong'), false, null, true);
                    }
                } catch (error) {
                    // Remove loading message
                    loadingDiv.remove();
                    
                    // Add error message
                    addMessage('Error: Unable to connect to the server', false, null, true);
                }
                
                // Re-enable button
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                
                // Focus back on input
                messageInput.focus();
            }
            
            async function loadChatHistory() {
                const userId = getCurrentUserId();
                const chatMessages = document.getElementById('chatMessages');
                
                try {
                    const response = await fetch(`/chat/history/${userId}`);
                    const data = await response.json();
                    
                    if (response.ok && data.chat_history && data.chat_history.length > 0) {
                        // Clear current messages
                        chatMessages.innerHTML = '';
                        
                        // Add system message about loaded history
                        addMessage(`Loaded ${data.chat_history.length} previous messages for User ${userId}`, false, null, false, true);
                        
                        // Add historical messages (reverse order since they come newest first)
                        data.chat_history.reverse().forEach(chat => {
                            addMessage(chat.message, true, chat.timestamp);
                            addMessage(chat.response, false, chat.timestamp);
                        });
                        
                        // Add welcome message
                        addMessage("History loaded! Continue your conversation below.", false, null, false, true);
                    } else {
                        addMessage(`No chat history found for User ${userId}`, false, null, false, true);
                    }
                } catch (error) {
                    addMessage('Error loading chat history', false, null, true);
                }
            }
            
            function clearChat() {
                const chatMessages = document.getElementById('chatMessages');
                chatMessages.innerHTML = `
                    <div class="message bot-message">
                        <div>Hello! I'm your AI assistant. Ask me anything!</div>
                        <div class="message-time">Just now</div>
                    </div>
                `;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            // Focus on input when page loads
            document.getElementById('messageInput').focus();
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/chat", response_model=ChatResponse)
async def chat_with_llm(chat_message: ChatMessage, db: Session = Depends(get_db)):
    """
    Single unified endpoint for chatting with the LLM using augmented retrieval.
    """
    try:
        # Get AI response using the LLM from Chatbot_retriver.py
        ai_response = augmented_retrieval(chat_message.message)
        
        # Log the chat interaction to database if user_id is provided
        if chat_message.user_id:
            try:
                chat_log = ChatbotLog(
                    user_id=chat_message.user_id,
                    message=chat_message.message,
                    response=ai_response
                )
                db.add(chat_log)
                db.commit()
            except Exception as log_error:
                # Log the exception but don't fail the chat response
                log_exception(log_error, "log_chat_interaction", 
                            {"user_id": chat_message.user_id, "message": chat_message.message}, db)
        
        return ChatResponse(response=ai_response, timestamp=datetime.utcnow())
        
    except Exception as e:
        # Log the exception
        log_exception(e, "chat_with_llm", {"message": chat_message.message}, db, chat_message.user_id)
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")

@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    """
    Get chat history for a specific user.
    """
    try:
        chat_logs = db.query(ChatbotLog).filter(
            ChatbotLog.user_id == user_id
        ).order_by(ChatbotLog.timestamp.desc()).limit(limit).all()
        
        return {
            "user_id": user_id,
            "chat_history": chat_logs,
            "count": len(chat_logs)
        }
    except Exception as e:
        log_exception(e, "get_chat_history", {"user_id": user_id}, db)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

# =============================================================================
# DOCUMENT UPLOAD ENDPOINT (Optional - for adding context to LLM)
# =============================================================================

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 200,
    db: Session = Depends(get_db)
):
    """
    Upload a document to provide context for the LLM responses.
    """
    try:
        # Validate file type
        allowed_extensions = {'.txt', '.md', '.csv', '.pdf'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Create file path
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create embeddings using DocumentLoaderManager
        try:
            result = DocumentLoaderManager.upload_and_create_embeddings(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_dir="Router/embedding",
                index_name="document_index"
            )
            
            if not result["success"]:
                raise Exception(result["error"])
            
            # Log to database
            embedding_record = Embedding(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                created_at=datetime.now()
            )
            db.add(embedding_record)
            db.commit()
            
            return DocumentUploadResponse(
                message=f"Document '{file.filename}' uploaded and embeddings created successfully!",
                file_path=file_path,
                filename=file.filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_id=embedding_record.id
            )
            
        except Exception as embedding_error:
            # Clean up uploaded file if embedding creation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create embeddings: {str(embedding_error)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        log_exception(str(e), "upload_document", db=db)
        # Clean up uploaded file if there's an error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# =============================================================================
# EXCEPTION LOGGING ENDPOINTS
# =============================================================================

@app.get("/api/exceptions")
async def get_exception_logs(limit: int = 50, db: Session = Depends(get_db)):
    """
    Get recent exception logs for monitoring and debugging.
    """
    try:
        exceptions = db.query(ExceptionLog).order_by(
            ExceptionLog.created_at.desc()
        ).limit(limit).all()
        return {"exception_logs": exceptions, "count": len(exceptions)}
    except Exception as e:
        log_exception(e, "get_exception_logs", None, db)
        raise HTTPException(status_code=500, detail="Failed to retrieve exception logs")

@app.get("/api/exceptions/{user_id}")
async def get_user_exceptions(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    """
    Get exception logs for a specific user.
    """
    try:
        exceptions = db.query(ExceptionLog).filter(
            ExceptionLog.user_id == user_id
        ).order_by(ExceptionLog.created_at.desc()).limit(limit).all()
        return {
            "user_id": user_id,
            "exception_logs": exceptions,
            "count": len(exceptions)
        }
    except Exception as e:
        log_exception(e, "get_user_exceptions", {"user_id": user_id}, db)
        raise HTTPException(status_code=500, detail="Failed to retrieve user exception logs")

@app.post("/api/exceptions/test")
async def test_exception_logging(db: Session = Depends(get_db)):
    """
    Test endpoint to demonstrate exception logging functionality.
    """
    try:
        # Intentionally cause an error for testing
        result = 1 / 0
        return {"result": result}
    except Exception as e:
        log_exception(e, "test_exception_logging", {"test": "intentional_error"}, db)
        return {"message": "Exception logged successfully", "error": str(e)}

@app.delete("/api/exceptions/cleanup")
async def cleanup_old_exceptions(days_old: int = 30, db: Session = Depends(get_db)):
    """
    Clean up exception logs older than specified days.
    """
    try:
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        deleted_count = db.query(ExceptionLog).filter(
            ExceptionLog.created_at < cutoff_date
        ).delete()
        db.commit()
        return {"message": f"Deleted {deleted_count} old exception logs"}
    except Exception as e:
        db.rollback()
        log_exception(e, "cleanup_exceptions", {"days_old": days_old}, db)
        raise HTTPException(status_code=500, detail="Failed to cleanup exception logs")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)





