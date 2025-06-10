from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from Router.table_creater import engine, get_db, Base
from Router.relations import ExceptionLog, ChatbotLog, Embedding
import uvicorn
import os
from pathlib import Path
import shutil

# Import Router modules for the three nodes
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
# PYDANTIC MODELS FOR THE THREE NODES
# =============================================================================

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime

class EmbeddingCreate(BaseModel):
    file_path: str
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 200

class FileUploadResponse(BaseModel):
    message: str
    file_path: str
    filename: str

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

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Immune File(Copy Hai JI💀💀)"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "nodes": ["chatbot", "embedding", "exception_logging"],
        "message": "All core nodes operational"
    }

# =============================================================================
# NODE 1: CHATBOT ENDPOINTS
# =============================================================================

@app.post("/api/chatbot/chat", response_model=ChatResponse)
async def chat_with_bot(chat_message: ChatMessage, db: Session = Depends(get_db)):
    """
    Chat with the AI assistant using retrieval augmentation.
    """
    try:
        # Use the chatbot retriever function from Router
        response = augmented_retrieval(chat_message.message)
        
        # Log the chat interaction to database
        if chat_message.user_id:
            try:
                chat_log = ChatbotLog(
                    user_id=chat_message.user_id,
                    message=chat_message.message,
                    response=response
                )
                db.add(chat_log)
                db.commit()
            except Exception as e:
                log_exception(e, "log_chat_interaction", {"user_id": chat_message.user_id}, db)
        
        return ChatResponse(response=response, timestamp=datetime.utcnow())
        
    except Exception as e:
        log_exception(e, "chat_endpoint", {"message": chat_message.message}, db, chat_message.user_id)
        raise HTTPException(status_code=500, detail="Failed to process chat message")

@app.get("/api/chatbot/history/{user_id}")
async def get_chat_history(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    """
    Get chat history for a specific user.
    """
    try:
        chat_logs = db.query(ChatbotLog).filter(
            ChatbotLog.user_id == user_id
        ).order_by(ChatbotLog.timestamp.desc()).limit(limit).all()
        return {"chat_history": chat_logs}
    except Exception as e:
        log_exception(e, "get_chat_history", {"user_id": user_id}, db)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

# =============================================================================
# NODE 2: EMBEDDING ENDPOINTS
# =============================================================================


@app.post("/api/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 200,
    db: Session = Depends(get_db)
):
    """
    Upload a document and automatically create embeddings for it.
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
# NODE 3: EXCEPTION LOGGING ENDPOINTS
# =============================================================================

@app.get("/api/exceptions")
async def get_exception_logs(limit: int = 50, db: Session = Depends(get_db)):
    """
    Get recent exception logs.
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
    Test endpoint to demonstrate exception logging.
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





