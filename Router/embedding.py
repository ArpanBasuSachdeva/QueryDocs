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
                                  index_name: str = None, chunk_size: int = 500, 
                                  chunk_overlap: int = 200, user_id: int = None):
        """Create embeddings from document and store them locally and in database."""
        # Ensure embedding directory exists
        os.makedirs(embedding_dir, exist_ok=True)
        
        # Process document to chunks
        chunks = DocumentLoaderManager.process_document_to_chunks(file_path, chunk_size, chunk_overlap)
        
        # Initialize embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Model name
            model_kwargs={'device': device},  # Device config
            encode_kwargs={'normalize_embeddings': True}  # Normalize
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create file hash for this document and chunking
        hash_code = DocumentLoaderManager.create_file_hash(file_path, chunk_size, chunk_overlap)
        # Use hash_code as index_name if not provided
        if index_name is None:
            index_name = hash_code
        
        # Save to local directory using hash_code as index_name
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
                                   index_name: str = None,
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
            
            # Always use hash_code as index_name
            hash_code = DocumentLoaderManager.create_file_hash(file_path, chunk_size, chunk_overlap)
            index_name = hash_code
            
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
                            "existing": True,
                            "hash_code": hash_code
                        }
                    }
                except:
                    print("Existing database record found but vector store missing, recreating...")
            
            # Create embeddings and save using hash_code as index_name
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
                    "existing": False,
                    "hash_code": hash_code
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
            # Extract actual embedding vectors and store them properly
            try:
                # Extract actual embedding vectors for the first few documents (to keep size manageable)
                sample_vectors = []  # Store sample vectors
                
                # Iterate through the index_to_docstore_id mapping to get vectors
                for vector_index, doc_id in list(vector_store.index_to_docstore_id.items())[:5]:  # Limit to first 5 vectors
                    vector = vector_store.index.reconstruct(vector_index).tolist()  # Extract vector from FAISS index
                    sample_vectors.append({  # Add vector data
                        "doc_id": doc_id,  # Document ID
                        "vector_index": vector_index,  # Vector index in FAISS
                        "vector": vector[:50]  # Store first 50 dimensions to keep size manageable
                    })
                
                embeddings_data = {  # Create embeddings data structure
                    "vector_count": len(vector_store.docstore._dict),  # Total vector count
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Model name
                    "chunk_size": chunk_size,  # Chunk size
                    "chunk_overlap": chunk_overlap,  # Chunk overlap
                    "file_name": Path(file_path).name,  # File name
                    "sample_vectors": sample_vectors,  # Sample embedding vectors
                    "vector_dimension": len(sample_vectors[0]["vector"]) if sample_vectors else 0,  # Vector dimension
                    "total_vector_dimension": vector_store.index.d if hasattr(vector_store.index, 'd') else 0  # Full vector dimension
                }
            except Exception as e:  # Handle extraction errors
                print(f"Warning: Could not extract embedding vectors: {e}")  # Log warning
                embeddings_data = {  # Fallback data structure
                    "vector_count": len(vector_store.docstore._dict) if hasattr(vector_store, 'docstore') else 0,  # Vector count
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Model name
                    "chunk_size": chunk_size,  # Chunk size
                    "chunk_overlap": chunk_overlap,  # Chunk overlap
                    "file_name": Path(file_path).name,  # File name
                    "error": str(e)  # Error message
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
            
            db.add(db_embedding)  # Add new embedding
            db.commit()  # Commit to DB
            db.refresh(db_embedding)  # Refresh to get ID

            # --- CLEANUP: Remove incomplete duplicates for this file_path ---
            db.query(EmbeddingModel).filter(
                EmbeddingModel.file_path == file_path,  # Same file_path
                EmbeddingModel.hash_code == None,       # No hash_code (incomplete)
                EmbeddingModel.embedding_vector == None, # No embedding_vector (incomplete)
                EmbeddingModel.id != db_embedding.id     # Exclude just-created record
            ).delete(synchronize_session=False)  # Delete incomplete duplicates
            db.commit()  # Commit cleanup
            # -------------------------------------------------------------

            print(f"✅ Embedding saved to database with ID: {db_embedding.id}")
            db.close()  # Close session
            return db_embedding  # Return new embedding
            
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

    @staticmethod
    def load_embeddings_by_hash(hash_code: str, embedding_dir: str = "Router/embedding"):
        """Load embedding metadata and vector store from the database and disk using a hash_code."""
        from Router.table_creater import SessionLocal  # Import here to avoid circular import
        from Router.relations import Embedding as EmbeddingModel
        import json
        import os
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Set device for embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Model name
            model_kwargs={'device': device},  # Device config
            encode_kwargs={'normalize_embeddings': True}  # Normalize
        )
        db = SessionLocal()  # Create DB session
        embedding = db.query(EmbeddingModel).filter(EmbeddingModel.hash_code == hash_code).first()  # Query by hash
        db.close()  # Close session
        if not embedding:
            return None, None  # Not found
        # Try to load the vector store from disk using the hash as index name
        index_name = hash_code  # Use hash_code as index name for storage
        vector_store_path = os.path.join(embedding_dir, index_name)
        if not os.path.exists(vector_store_path):
            return None, embedding  # Metadata found, but no vector store
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)  # Load vector store
        return vector_store, embedding  # Return both

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



