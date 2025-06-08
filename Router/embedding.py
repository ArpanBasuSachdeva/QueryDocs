# !pip install -q youtube-transcript-api langchain-community langchain-openai \
#                faiss-cpu tiktoken python-dotenv sentence-transformers torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from Router.exception_utils import log_exception
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    model_kwargs={'device': device},  
    encode_kwargs={'normalize_embeddings': True}  
)

def store_embeddings_in_chroma(text, persist_directory="./chroma_db"):
    try:
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=200  
        )
        
        
        chunks = splitter.create_documents([text])  
        
        
        vector_store = Chroma.from_documents(
            documents=chunks,  
            embedding=embeddings,  
            persist_directory=persist_directory  
        )
        
        
        vector_store.persist()  
        
        return vector_store  
        
    except Exception as e:
        log_exception(e)  
        raise 



