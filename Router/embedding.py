from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
from Router.exception_utils import log_exception
import os
import json
import hashlib
from pathlib import Path
from typing import List
from Router.table_creater import SessionLocal
from Router.relations import Embedding as EmbeddingModel

class DocumentLoaderManager:
    loader_map = {
        '.txt': TextLoader,
        '.md': TextLoader,
        '.csv': CSVLoader,
        '.pdf': PyPDFLoader,
    }

    @staticmethod
    def load(file_path: str) -> List[Document]:
        suffix = Path(file_path).suffix.lower()
        loader_cls = DocumentLoaderManager.loader_map.get(suffix)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {suffix}")
        loader = loader_cls(file_path)
        return loader.load()
    
    @staticmethod
    def load_and_join_content(file_path: str) -> str:
        """Load documents from file and join their content into a single string."""
        docs = DocumentLoaderManager.load(file_path)
        transcript = " ".join(chunk.page_content for chunk in docs)
        return transcript
    
    @staticmethod
    def process_document_to_chunks(file_path: str, chunk_size: int = 500, chunk_overlap: int = 200):
        """Load document, join content, and split into chunks."""
        transcript = DocumentLoaderManager.load_and_join_content(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.create_documents([transcript])
        return chunks
    
    @staticmethod
    def create_and_store_embeddings(file_path: str, embedding_dir: str = "Router/embedding", 
                                  index_name: str = "document_index", chunk_size: int = 500, 
                                  chunk_overlap: int = 200, user_id: int = None):
        """Create embeddings from document and store them locally and in database."""
        # Ensure embedding directory exists
        os.makedirs(embedding_dir, exist_ok=True)
        
        # Process document to chunks
        chunks = DocumentLoaderManager.process_document_to_chunks(file_path, chunk_size, chunk_overlap)
        
        # Initialize embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  
            model_kwargs={'device': device},  
            encode_kwargs={'normalize_embeddings': True}  
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Save to local directory
        vector_store.save_local(os.path.join(embedding_dir, index_name))
        
        # Save to database
        db_embedding = DocumentLoaderManager.save_embedding_to_db(
            file_path=file_path,
            vector_store=vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            user_id=user_id
        )
        
        print(f"Embeddings saved to {embedding_dir}/{index_name}")
        if db_embedding:
            print(f"Embeddings metadata saved to database with ID: {db_embedding.id}")
        
        return vector_store
    
    @staticmethod
    def load_embeddings(embedding_dir: str = "Router/embedding", index_name: str = "document_index"):
        """Load embeddings from local storage."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  
            model_kwargs={'device': device},  
            encode_kwargs={'normalize_embeddings': True}  
        )
        
        vector_store = FAISS.load_local(
            os.path.join(embedding_dir, index_name), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    
    @staticmethod
    def upload_and_create_embeddings(file_path: str, chunk_size: int = 500, 
                                   chunk_overlap: int = 200, 
                                   embedding_dir: str = "Router/embedding",
                                   index_name: str = "document_index",
                                   user_id: int = None):
        """
        Complete workflow: validate file, create embeddings, and return metadata.
        This method is specifically designed for API upload endpoints.
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Validate file type
            allowed_extensions = {'.txt', '.md', '.csv', '.pdf'}
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension not in allowed_extensions:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Check if embedding already exists in database
            existing_embedding = DocumentLoaderManager.get_embedding_from_db(
                file_path, chunk_size, chunk_overlap
            )
            
            if existing_embedding:
                print(f"Embedding already exists in database for this file")
                # Load existing vector store
                try:
                    vector_store = DocumentLoaderManager.load_embeddings(embedding_dir, index_name)
                    return {
                        "success": True,
                        "vector_store": vector_store,
                        "file_info": {
                            "filename": existing_embedding.file_path,
                            "file_path": file_path,
                            "file_size": os.path.getsize(file_path),
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "db_id": existing_embedding.id,
                            "existing": True
                        }
                    }
                except:
                    print("Existing database record found but vector store missing, recreating...")
            
            # Create embeddings
            vector_store = DocumentLoaderManager.create_and_store_embeddings(
                file_path=file_path,
                embedding_dir=embedding_dir,
                index_name=index_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                user_id=user_id
            )
            
            # Get file info
            file_size = os.path.getsize(file_path)
            filename = Path(file_path).name
            
            # Get database record
            db_embedding = DocumentLoaderManager.get_embedding_from_db(
                file_path, chunk_size, chunk_overlap
            )
            
            return {
                "success": True,
                "vector_store": vector_store,
                "file_info": {
                    "filename": filename,
                    "file_path": file_path,
                    "file_size": file_size,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "db_id": db_embedding.id if db_embedding else None,
                    "existing": False
                }
            }
            
        except Exception as e:
            # Pass the database session if available
            db_session = locals().get('db')
            log_exception(str(e), "upload_and_create_embeddings", db=db_session)
            return {
                "success": False,
                "error": str(e),
                "file_info": {
                    "filename": Path(file_path).name if file_path else "Unknown",
                    "file_path": file_path,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            }

    @staticmethod
    def create_file_hash(file_path: str, chunk_size: int = 500, chunk_overlap: int = 200) -> str:
        """Create a unique hash for the file based on content and parameters."""
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Include chunk parameters in hash to differentiate between different processing settings
        hash_input = file_content + f"{chunk_size}_{chunk_overlap}".encode()
        return hashlib.md5(hash_input).hexdigest()
    
    @staticmethod
    def save_embedding_to_db(file_path: str, vector_store, chunk_size: int = 500, 
                           chunk_overlap: int = 200, user_id: int = None):
        """Save embedding information to PostgreSQL database."""
        try:
            db = SessionLocal()
            
            # Create file hash
            hash_code = DocumentLoaderManager.create_file_hash(file_path, chunk_size, chunk_overlap)
            
            # Check if embedding already exists
            existing = db.query(EmbeddingModel).filter(EmbeddingModel.hash_code == hash_code).first()
            if existing:
                print(f"Embedding already exists in database for hash: {hash_code}")
                db.close()
                return existing
            
            # Get embedding vectors from vector store
            # For demonstration, we'll store the first few embedding vectors as JSON
            # In production, you might want to store all vectors or use a vector database
            embeddings_data = {
                "vector_count": len(vector_store.docstore._dict),
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "file_name": Path(file_path).name
            }
            
            # Create new embedding record (user_id can be None)
            db_embedding = EmbeddingModel(
                file_path=file_path,
                hash_code=hash_code,
                embedding_vector=json.dumps(embeddings_data),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                user_id=user_id if user_id else None  # Allow None user_id
            )
            
            db.add(db_embedding)
            db.commit()
            db.refresh(db_embedding)
            
            print(f"✅ Embedding saved to database with ID: {db_embedding.id}")
            db.close()
            return db_embedding
            
        except Exception as e:
            db_session = locals().get('db')
            log_exception(str(e), "save_embedding_to_db", db=db_session)
            if 'db' in locals():
                db.close()
            return None
    
    @staticmethod
    def get_embedding_from_db(file_path: str, chunk_size: int = 500, chunk_overlap: int = 200):
        """Retrieve embedding information from PostgreSQL database."""
        try:
            db = SessionLocal()
            
            # Create file hash
            hash_code = DocumentLoaderManager.create_file_hash(file_path, chunk_size, chunk_overlap)
            
            # Query database
            embedding = db.query(EmbeddingModel).filter(EmbeddingModel.hash_code == hash_code).first()
            
            db.close()
            return embedding
            
        except Exception as e:
            db_session = locals().get('db')
            log_exception(str(e), "get_embedding_from_db", db=db_session)
            if 'db' in locals():
                db.close()
            return None

device = "cuda" if torch.cuda.is_available() else "cpu"

# Example usage:
# file_path = "path/to/your/document.txt"
# vector_store = DocumentLoaderManager.create_and_store_embeddings(file_path)
# 
# # To load existing embeddings:
# vector_store = DocumentLoaderManager.load_embeddings()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    model_kwargs={'device': device},  
    encode_kwargs={'normalize_embeddings': True}  
)



