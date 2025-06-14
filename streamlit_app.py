import streamlit as st
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import json
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import tempfile
import shutil

# Constants
UPLOAD_DIR = "uploads"
VECTOR_STORE_DIR = "vector_store"

# Create directories if they don't exist
upload_dir = Path(UPLOAD_DIR)
upload_dir.mkdir(exist_ok=True)
vector_store_dir = Path(VECTOR_STORE_DIR)
vector_store_dir.mkdir(exist_ok=True)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Page config
st.set_page_config(
    page_title="QueryDocs",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'selected_document_hash' not in st.session_state:
    st.session_state.selected_document_hash = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Helper functions
def get_documents():
    return st.session_state.documents

def save_document_info(filename, hash_code, file_size, status="processed"):
    doc_info = {
        "filename": filename,
        "hash_code": hash_code,
        "file_size": file_size,
        "status": status,
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
    st.session_state.documents.append(doc_info)
    return doc_info

def process_document(file, chunk_size, chunk_overlap):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        # Read the file content
        with open(tmp_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)

        # Generate hash code for the document
        hash_code = str(uuid.uuid4())

        # Create and save vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(f"{VECTOR_STORE_DIR}/{hash_code}")

        # Save document info
        doc_info = save_document_info(
            filename=file.name,
            hash_code=hash_code,
            file_size=len(file.getvalue())
        )

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return True, doc_info
    except Exception as e:
        return False, str(e)

def query_document(message, hash_code):
    try:
        # Load vector store
        vector_store = FAISS.load_local(f"{VECTOR_STORE_DIR}/{hash_code}", embeddings)
        
        # Perform similarity search
        docs = vector_store.similarity_search(message, k=3)
        
        # Combine relevant chunks
        context = "\n".join([doc.page_content for doc in docs])
        
        # Here you would typically use an LLM to generate a response
        # For now, we'll return the context as the response
        response = f"Based on the document context:\n\n{context}"
        
        return True, {"response": response}
    except Exception as e:
        return False, str(e)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Upload Document", "Documents", "Chat History"])

# Display session ID in sidebar
st.sidebar.markdown("---")
st.sidebar.info(f"Session ID: {st.session_state.session_id}")

# Main content
if page == "Chat":
    st.title("🤖 QueryDocs")
    
    # Document selection section
    st.subheader("Select Document")
    documents = get_documents()
    
    if documents:
        # Create a dictionary of filename to hash_code
        doc_options = {f"{doc['filename']} ({doc['status']})": doc['hash_code'] 
                     for doc in documents if doc['hash_code']}
        
        selected_doc = st.selectbox(
            "Select a document to chat with:",
            options=list(doc_options.keys())
        )
        st.session_state.selected_document_hash = doc_options[selected_doc]
        
        # Show selected document info
        selected_doc_info = next((doc for doc in documents if doc['hash_code'] == st.session_state.selected_document_hash), None)
        if selected_doc_info:
            st.info(f"Chatting with document: {selected_doc_info['filename']}")
            
            # Chat interface
            st.subheader("Chat Interface")
            user_message = st.text_input("Enter your message:")
            if st.button("Send"):
                if user_message:
                    success, response = query_document(
                        message=user_message,
                        hash_code=st.session_state.selected_document_hash
                    )
                    if success:
                        st.success("Message sent successfully!")
                        # Update session chat history
                        st.session_state.chat_history.append({
                            "message": user_message,
                            "response": response["response"],
                            "timestamp": datetime.now().isoformat(),
                            "hash_code": st.session_state.selected_document_hash
                        })
                        st.write("**AI Response:**", response["response"])
                    else:
                        st.error(response)
            
            # Display current session chat history
            if st.session_state.chat_history:
                st.subheader("Current Session Chat History")
                for chat in reversed(st.session_state.chat_history):
                    if chat['hash_code'] == st.session_state.selected_document_hash:
                        with st.expander(f"Chat at {chat['timestamp']}"):
                            st.write(f"**User:** {chat['message']}")
                            st.write(f"**AI:** {chat['response']}")
    else:
        st.warning("No documents available. Please upload a document first.")

elif page == "Upload Document":
    st.title("📤 Upload Document")
    
    # Upload section
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'md', 'csv', 'pdf'])
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=1000, value=500)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=50, max_value=500, value=200)
    
    if uploaded_file and st.button("Upload"):
        success, response = process_document(uploaded_file, chunk_size, chunk_overlap)
        if success:
            st.success("Document uploaded successfully!")
            st.json(response)
        else:
            st.error(response)

elif page == "Documents":
    st.title("📚 Uploaded Documents")
    
    # Document list
    documents = get_documents()
    if documents:
        # Convert to DataFrame for better display
        df = pd.DataFrame(documents)
        # Format the DataFrame
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['file_size'] = df['file_size'].apply(lambda x: f"{x/1024:.2f} KB" if x else "N/A")
        
        # Display the DataFrame
        st.dataframe(
            df[['filename', 'file_size', 'status', 'created_at', 'is_active', 'hash_code']],
            use_container_width=True
        )
        
        # Show detailed view in expandable sections
        for doc in documents:
            with st.expander(f"Details: {doc['filename']}"):
                st.json(doc)
    else:
        st.info("No documents uploaded yet")

elif page == "Chat History":
    st.title("💬 Chat History")
    
    # Display chat history
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            with st.expander(f"Chat at {chat['timestamp']}"):
                st.write(f"**User:** {chat['message']}")
                st.write(f"**AI:** {chat['response']}")
                if chat.get('hash_code'):
                    st.write(f"**Document Hash:** {chat['hash_code']}")
                if chat.get('session_id'):
                    st.write(f"**Session ID:** {chat['session_id']}")
    else:
        st.info("No chat history available") 
