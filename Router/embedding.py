from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
from Router.exception_utils import log_exception
import os
from pathlib import Path
from typing import List

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
                                  chunk_overlap: int = 200):
        """Create embeddings from document and store them locally."""
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
        
        print(f"Embeddings saved to {embedding_dir}/{index_name}")
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



