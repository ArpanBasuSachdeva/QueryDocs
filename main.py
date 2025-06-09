from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from functioncaller.database import engine, get_db
from functioncaller import model
import uvicorn

# Import Router modules for the three nodes
from Router.Chatbot_retriver import augmented_retrieval
from Router.exception_utils import log_exception
from Router.embedding import DocumentLoaderManager
from Router.relations import ExceptionLog, ChatbotLog, Embedding
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Create database tables
model.Base.metadata.create_all(bind=engine)

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

# =============================================================================
# ROOT AND HEALTH ENDPOINTS
# =============================================================================

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Immune File(Copy Hai JI💀💀)"}

@app.get("/upload", response_class=HTMLResponse)
async def upload_form():
    """
    Get chat history for a specific user.
    """
    return {"message": "Welcome to the Immune File(Copy Hai JI💀💀)"}


@app.get("/ee")
async def offline_game():
    """
    Offline skull game - like the Chrome dinosaur game but with skull emoji
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Skull Runner Game</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                background: #f7f7f7;
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            #gameContainer {
                text-align: center;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            #gameCanvas {
                border: 2px solid #535353;
                background: #f7f7f7;
            }
            #score {
                font-size: 24px;
                margin-bottom: 10px;
                color: #535353;
            }
            #instructions {
                margin-top: 10px;
                color: #8b8b8b;
            }
        </style>
    </head>
    <body>
        <div id="gameContainer">
            <div id="score">Score: 0</div>
            <canvas id="gameCanvas" width="800" height="200"></canvas>
            <div id="instructions">Press SPACE or click to jump!</div>
        </div>

        <script>
            const canvas = document.getElementById('gameCanvas');
            const ctx = canvas.getContext('2d');
            const scoreElement = document.getElementById('score');

            // Game variables
            let gameSpeed = 3;
            let gravity = 0.5;
            let score = 0;
            let gameRunning = true;

            // Skull player
            const skull = {
                x: 50,
                y: 150,
                width: 40,
                height: 40,
                dy: 0,
                jumpPower: 12,
                grounded: false,
                emoji: '💀'
            };

            // Obstacles array
            const obstacles = [];

            // Ground
            const ground = {
                x: 0,
                y: canvas.height - 20,
                width: canvas.width,
                height: 20
            };

            // Input handling
            document.addEventListener('keydown', (e) => {
                if (e.code === 'Space' && gameRunning) {
                    e.preventDefault();
                    jump();
                }
            });

            canvas.addEventListener('click', () => {
                if (gameRunning) {
                    jump();
                }
            });

            function jump() {
                if (skull.grounded) {
                    skull.dy = -skull.jumpPower;
                    skull.grounded = false;
                }
            }

            function createObstacle() {
                const obstacle = {
                    x: canvas.width,
                    y: ground.y - 30,
                    width: 20,
                    height: 30
                };
                obstacles.push(obstacle);
            }

            function updateSkull() {
                // Apply gravity
                skull.dy += gravity;
                skull.y += skull.dy;

                // Ground collision
                if (skull.y + skull.height >= ground.y) {
                    skull.y = ground.y - skull.height;
                    skull.dy = 0;
                    skull.grounded = true;
                }
            }

            function updateObstacles() {
                // Move obstacles
                for (let i = obstacles.length - 1; i >= 0; i--) {
                    obstacles[i].x -= gameSpeed;

                    // Remove obstacles that are off screen
                    if (obstacles[i].x + obstacles[i].width < 0) {
                        obstacles.splice(i, 1);
                        score += 10;
                        scoreElement.textContent = 'Score: ' + score;
                        
                        // Increase game speed gradually
                        if (score % 100 === 0) {
                            gameSpeed += 0.5;
                        }
                    }
                }

                // Create new obstacles
                if (Math.random() < 0.005) {
                    createObstacle();
                }
            }

            function checkCollisions() {
                for (let obstacle of obstacles) {
                    if (skull.x < obstacle.x + obstacle.width &&
                        skull.x + skull.width > obstacle.x &&
                        skull.y < obstacle.y + obstacle.height &&
                        skull.y + skull.height > obstacle.y) {
                        gameRunning = false;
                        return true;
                    }
                }
                return false;
            }

            function draw() {
                // Clear canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw ground
                ctx.fillStyle = '#535353';
                ctx.fillRect(ground.x, ground.y, ground.width, ground.height);

                // Draw skull player
                ctx.font = '40px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(skull.emoji, skull.x + skull.width/2, skull.y + skull.height - 5);

                // Draw obstacles
                ctx.fillStyle = '#535353';
                for (let obstacle of obstacles) {
                    ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
                }

                // Game over screen
                if (!gameRunning) {
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    ctx.fillStyle = 'white';
                    ctx.font = '40px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('Game Over!', canvas.width/2, canvas.height/2 - 20);
                    ctx.font = '20px Arial';
                    ctx.fillText('Press F5 to restart', canvas.width/2, canvas.height/2 + 20);
                }
            }

            function gameLoop() {
                if (gameRunning) {
                    updateSkull();
                    updateObstacles();
                    checkCollisions();
                }
                
                draw();
                requestAnimationFrame(gameLoop);
            }

            // Start the game
            gameLoop();
        </script>
    </body>
    </html>
    """



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

@app.post("/api/embeddings")
async def create_embeddings(embedding_data: EmbeddingCreate, db: Session = Depends(get_db)):
    """
    Create embeddings from a document file.
    """
    try:
        # Use the DocumentLoaderManager to create embeddings
        vector_store = DocumentLoaderManager.create_and_store_embeddings(
            file_path=embedding_data.file_path,
            chunk_size=embedding_data.chunk_size,
            chunk_overlap=embedding_data.chunk_overlap
        )
        
        return {
            "message": "Embeddings created successfully",
            "file_path": embedding_data.file_path,
            "status": "completed"
        }
    except Exception as e:
        log_exception(e, "create_embeddings", {"file_path": embedding_data.file_path}, db)
        raise HTTPException(status_code=500, detail="Failed to create embeddings")

@app.get("/api/embeddings")
async def get_embeddings(db: Session = Depends(get_db)):
    """
    Get all embeddings from the database.
    """
    try:
        embeddings = db.query(Embedding).all()
        return {"embeddings": embeddings}
    except Exception as e:
        log_exception(e, "get_embeddings", None, db)
        raise HTTPException(status_code=500, detail="Failed to retrieve embeddings")

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





