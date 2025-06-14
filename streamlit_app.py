import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import json
import uuid

# Constants
API_BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:8000')  # Use environment variable with local fallback
UPLOAD_DIR = "CopyHaiJi//uploads"

# Create directories if they don't exist
base_dir = Path("CopyHaiJi")
base_dir.mkdir(exist_ok=True)
upload_dir = base_dir / "uploads"
upload_dir.mkdir(exist_ok=True)

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

# Helper functions
def get_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            return response.json()["documents"]
        st.error(f"Failed to fetch documents: {response.status_code}")
        return []
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def get_chat_history():
    try:
        response = requests.get(f"{API_BASE_URL}/chat/history")
        if response.status_code == 200:
            return response.json()["chat_history"]
        st.error(f"Failed to fetch chat history: {response.status_code}")
        return []
    except Exception as e:
        st.error(f"Error fetching chat history: {str(e)}")
        return []

def get_exceptions():
    try:
        response = requests.get(f"{API_BASE_URL}/api/exceptions/table")
        if response.status_code == 200:
            return response.text
        st.error(f"Failed to fetch exceptions: {response.status_code}")
        return ""
    except Exception as e:
        st.error(f"Error fetching exceptions: {str(e)}")
        return ""

def send_chat_message(message, hash_code):
    try:
        payload = json.dumps({
            "message": message,
            "hash_code": hash_code,
            "session_id": st.session_state.session_id
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            f"{API_BASE_URL}/chat",
            data=payload,
            headers=headers
        )
        if response.status_code == 200:
            return True, response.json()
        return False, f"Failed to send message: {response.status_code}"
    except Exception as e:
        return False, f"Error sending message: {str(e)}"

def upload_document(file, chunk_size, chunk_overlap):
    try:
        files = {"file": (file.name, file.getvalue())}
        params = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "session_id": st.session_state.session_id
        }
        response = requests.post(
            f"{API_BASE_URL}/upload-document",
            files=files,
            params=params
        )
        if response.status_code == 200:
            return True, response.json()
        return False, f"Failed to upload document: {response.status_code}"
    except Exception as e:
        return False, f"Error uploading document: {str(e)}"

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Upload Document", "Documents", "Chat History", "Exceptions"])

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
                    success, response = send_chat_message(
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
        success, response = upload_document(uploaded_file, chunk_size, chunk_overlap)
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
    chat_history = get_chat_history()
    if chat_history:
        for chat in chat_history:
            with st.expander(f"Chat at {chat['timestamp']}"):
                st.write(f"**User:** {chat['message']}")
                st.write(f"**AI:** {chat['response']}")
                if chat.get('hash_code'):
                    st.write(f"**Document Hash:** {chat['hash_code']}")
                if chat.get('session_id'):
                    st.write(f"**Session ID:** {chat['session_id']}")
    else:
        st.info("No chat history available")

elif page == "Exceptions":
    st.title("⚠️ Exception Logs")
    
    # Exception table
    exceptions_html = get_exceptions()
    if exceptions_html:
        st.components.v1.html(exceptions_html, height=600)
    else:
        st.info("No exception logs available") 
