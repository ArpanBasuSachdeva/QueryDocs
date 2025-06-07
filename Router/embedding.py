# !pip install -q youtube-transcript-api langchain-community langchain-openai \
#                faiss-cpu tiktoken python-dotenv sentence-transformers torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import  Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from Router.exception_utils import log_exception

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize embeddings with CUDA support
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([input_text])

# Initialize Chroma vector store
def create_chroma_store(chunks, persist_directory="./chroma_db"):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database to disk
    vector_store.persist()
    
    return vector_store

